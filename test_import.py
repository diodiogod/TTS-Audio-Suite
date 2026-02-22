import sys
try:
    print("Pretending to fail import...")
    raise Exception("Boom")
except Exception:
    pass

import librosa
try:
    librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)
except Exception as e:
    print(f"Test failed with: {e}")

print("Testing subsequent import...")
try:
    from librosa.filters import mel
    print("Success!")
except Exception as e:
    print(f"Subsequent import failed with: {type(e).__name__}: {e}")
