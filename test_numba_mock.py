import numba
from numba.core.typing.templates import AbstractTemplate
from numba.core.extending import type_callable

def dummy_guvectorize(*args, **kwargs):
    def decorator(func):
        # Numba's type inference expects the decorated object
        # to either be a Numba Dispatcher or something that provides get_call_template
        # Let's try wrapping it in a mock object
        class MockGufunc:
            def __init__(self, f):
                self.f = f
                self._is_numba_cfunc = True
            
            def __call__(self, *a, **kw):
                return self.f(*a, **kw)
                
            def get_call_template(self, *a, **k):
                pass
                
        return MockGufunc(func)
    return decorator

numba.guvectorize = dummy_guvectorize

try:
    import librosa
    print("Librosa imported successfully!")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
