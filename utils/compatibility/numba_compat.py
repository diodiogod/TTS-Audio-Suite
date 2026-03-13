"""
Centralized Numba/Librosa Compatibility System
Fast startup testing with intelligent JIT fallback management
"""

import sys
import os
from typing import Optional, Dict, Any
import time
from importlib.metadata import PackageNotFoundError, version as package_version


def _parse_version_tuple(version_text: str) -> tuple:
    parts = []
    for piece in version_text.split("."):
        digits = ""
        for char in piece:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _get_installed_numba_version() -> Optional[str]:
    try:
        return package_version("numba")
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _is_python313_numba_disable_jit_risky() -> bool:
    """Known-bad combo: Python 3.13 + numba 0.64+ with NUMBA_DISABLE_JIT."""
    if sys.version_info < (3, 13):
        return False

    numba_version = _get_installed_numba_version()
    if not numba_version:
        return False

    return _parse_version_tuple(numba_version) >= (0, 64)

class NumbaCompatibilityManager:
    """
    Smart numba compatibility system that tests JIT compilation at startup
    and conditionally applies workarounds only when needed.
    """
    
    def __init__(self):
        self._jit_disabled = False
        self._test_results = {}
        self._startup_time = None
        
    def test_numba_compatibility(self, quick_test: bool = True) -> Dict[str, Any]:
        """
        Test numba JIT compilation compatibility.
        Returns dict with test results and timing info.

        When quick_test=True (default at startup), we skip the actual librosa import
        to avoid the ~0.7s cost. We do not assume Python 3.13 is incompatible anymore:
        some newer numba/librosa stacks work correctly, and forcing NUMBA_DISABLE_JIT
        can itself cause get_call_template failures on Python 3.13 + numba 0.64+.
        """
        start_time = time.time()
        results = {
            'jit_compatible': True,
            'test_duration': 0,
            'errors': [],
            'environment_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'is_python_313': sys.version_info >= (3, 13),
                'numba_version': _get_installed_numba_version(),
            }
        }

        # FAST PATH: skip actual librosa import during startup to save ~0.7s.
        # We no longer treat Python 3.13 as automatically broken.
        if quick_test:
            results['test_duration'] = time.time() - start_time
            self._test_results = results
            return results

        # Test librosa.resample — this lazy-loads librosa.core.audio which contains
        # the @guvectorize decorated function that crashes on NumPy 2.x + certain hardware.
        # librosa.stft does NOT trigger this code path, so we must test resample directly.
        try:
            import numpy as np
            import librosa

            test_audio = np.random.randn(1024).astype(np.float32)
            _ = librosa.resample(y=test_audio, orig_sr=22050, target_sr=16000)

        except Exception as e:
            error_msg = str(e)
            if "'function' object has no attribute 'get_call_template'" in error_msg or \
               "Cannot determine Numba type" in error_msg:
                results['jit_compatible'] = False
                results['errors'].append(f"librosa.resample JIT failure: {error_msg}")
            else:
                results['errors'].append(f"librosa test error: {error_msg}")
        
        results['test_duration'] = time.time() - start_time
        self._test_results = results
        return results
    
    def apply_jit_workaround(self) -> None:
        """Apply numba JIT disabling workaround."""
        if self._jit_disabled:
            return  # Already applied

        if _is_python313_numba_disable_jit_risky():
            print("⚠️ Skipping NUMBA_DISABLE_JIT workaround on Python 3.13 + numba 0.64+ because it can break librosa imports")
            return
            
        # Set environment variable
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        os.environ['NUMBA_ENABLE_CUDASIM'] = '1'  # Additional compatibility
        
        # Also set numba config if already imported
        try:
            import numba
            numba.config.DISABLE_JIT = True
            numba.config.ENABLE_CUDASIM = True
        except ImportError:
            pass  # numba not imported yet, environment variable will handle it

        self._jit_disabled = True
        print("🔧 Applied numba JIT workaround for Python 3.12+/Numpy 2.x compatibility")
    
    def setup_smart_compatibility(self, quick_startup: bool = True) -> Dict[str, Any]:
        """
        Main entry point: Test compatibility and apply workarounds if needed.
        
        Args:
            quick_startup: If True, uses faster test (default). False = thorough test.
        
        Returns:
            Dict with setup results and timing info
        """
        setup_start = time.time()
        
        # If JIT already disabled by user/environment, still apply numba.config
        # because numba may already be imported and the env var alone won't take effect.
        if os.environ.get('NUMBA_DISABLE_JIT') == '1':
            if _is_python313_numba_disable_jit_risky():
                return {
                    'status': 'jit_disable_skipped',
                    'message': '⚠️ Leaving NUMBA_DISABLE_JIT compatibility untouched on Python 3.13 + numba 0.64+ (known risky combo)',
                    'setup_time': time.time() - setup_start
                }
            try:
                import numba
                numba.config.DISABLE_JIT = True
            except Exception:
                pass
            return {
                'status': 'jit_already_disabled',
                'message': 'NUMBA_DISABLE_JIT already set by user/environment',
                'setup_time': time.time() - setup_start
            }
        
        # Test compatibility (fast by default)
        test_results = self.test_numba_compatibility(quick_test=quick_startup)
        
        setup_time = time.time() - setup_start
        self._startup_time = setup_time
        
        if test_results['jit_compatible']:
            return {
                'status': 'compatible',
                'message': f'✅ Numba JIT working properly (tested in {test_results["test_duration"]:.2f}s)',
                'setup_time': setup_time,
                'test_results': test_results
            }
        else:
            # Apply workaround
            self.apply_jit_workaround()
            
            return {
                'status': 'workaround_applied', 
                'message': f'🔧 Applied JIT workaround due to compatibility issues (tested in {test_results["test_duration"]:.2f}s)',
                'setup_time': setup_time,
                'test_results': test_results,
                'errors': test_results['errors']
            }
    
    def is_jit_disabled(self) -> bool:
        """Check if JIT has been disabled by this manager."""
        return self._jit_disabled
    
    def get_compatibility_status(self) -> Dict[str, Any]:
        """Get current compatibility status and test results."""
        return {
            'jit_disabled': self._jit_disabled,
            'test_results': self._test_results,
            'startup_time': self._startup_time,
            'environment': {
                'numba_disable_jit': os.environ.get('NUMBA_DISABLE_JIT'),
                'numba_enable_cudasim': os.environ.get('NUMBA_ENABLE_CUDASIM'),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
        }


# Global instance
_numba_manager = NumbaCompatibilityManager()

def setup_numba_compatibility(quick_startup: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """
    Main function to set up numba compatibility.
    Call this once at application startup.
    
    Args:
        quick_startup: Use fast compatibility test (default: True)
        verbose: Print setup messages (default: True)
    
    Returns:
        Dict with setup results
    """
    results = _numba_manager.setup_smart_compatibility(quick_startup=quick_startup)
    
    if verbose and results['setup_time'] > 0.1:  # Only show timing for slow setups
        print(f"🔬 Numba compatibility setup: {results['setup_time']:.2f}s")
    
    if verbose:
        print(results['message'])
        
        # Show any errors in verbose mode
        if 'errors' in results and results['errors']:
            print("   Compatibility errors detected:")
            for error in results['errors'][:2]:  # Show first 2 errors
                print(f"   • {error}")
    
    return results

def is_jit_disabled() -> bool:
    """Check if JIT has been disabled by the compatibility system."""
    return _numba_manager.is_jit_disabled()

def get_compatibility_status() -> Dict[str, Any]:
    """Get detailed compatibility status for debugging."""
    return _numba_manager.get_compatibility_status()

def force_apply_workaround() -> None:
    """Force apply JIT workaround without testing (for emergency fallback)."""
    _numba_manager.apply_jit_workaround()
