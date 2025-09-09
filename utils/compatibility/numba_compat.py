"""
Centralized Numba/Librosa Compatibility System
Fast startup testing with intelligent JIT fallback management
"""

import sys
import os
import warnings
from typing import Optional, Dict, Any
import time

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
        """
        start_time = time.time()
        results = {
            'jit_compatible': True,
            'test_duration': 0,
            'errors': [],
            'environment_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'is_python_313': sys.version_info >= (3, 13)
            }
        }
        
        # Test 1: Basic librosa.stft compilation (most common failure point)
        try:
            import numpy as np
            import librosa
            
            # Create minimal test signal
            test_audio = np.random.randn(1024).astype(np.float32)
            
            # This will trigger numba JIT compilation if enabled
            _ = librosa.stft(test_audio, hop_length=256, n_fft=512)
            
        except Exception as e:
            error_msg = str(e)
            if "'function' object has no attribute 'get_call_template'" in error_msg:
                results['jit_compatible'] = False
                results['errors'].append(f"librosa.stft JIT failure: {error_msg}")
            elif "Cannot determine Numba type" in error_msg:
                results['jit_compatible'] = False  
                results['errors'].append(f"Numba type inference failure: {error_msg}")
            else:
                # Other errors might not be JIT-related
                results['errors'].append(f"librosa test error: {error_msg}")
        
        if not quick_test and results['jit_compatible']:
            # Test 2: librosa.filters.mel (secondary failure point)
            try:
                _ = librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)
            except Exception as e:
                error_msg = str(e)
                if "get_call_template" in error_msg or "Cannot determine Numba type" in error_msg:
                    results['jit_compatible'] = False
                    results['errors'].append(f"librosa.filters.mel JIT failure: {error_msg}")
                else:
                    results['errors'].append(f"librosa.filters.mel error: {error_msg}")
        
        results['test_duration'] = time.time() - start_time
        self._test_results = results
        return results
    
    def apply_jit_workaround(self) -> None:
        """Apply numba JIT disabling workaround."""
        if self._jit_disabled:
            return  # Already applied
            
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
        print("🔧 Applied numba JIT workaround for Python 3.13 compatibility")
    
    def setup_smart_compatibility(self, quick_startup: bool = True) -> Dict[str, Any]:
        """
        Main entry point: Test compatibility and apply workarounds if needed.
        
        Args:
            quick_startup: If True, uses faster test (default). False = thorough test.
        
        Returns:
            Dict with setup results and timing info
        """
        setup_start = time.time()
        
        # Skip testing if JIT already disabled by user/environment
        if os.environ.get('NUMBA_DISABLE_JIT') == '1':
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