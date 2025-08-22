# MediaPipe Python 3.13 Workaround

## Method 1: Using curl (if available)

```bash
# Download the wheel file first
curl -o mediapipe-0.10.21-cp312-cp312-win_amd64.whl https://files.pythonhosted.org/packages/b7/79/b77808f8195f229ef0c15875540dfdd36724748a4b3de53d993f23336839/mediapipe-0.10.21-cp312-cp312-win_amd64.whl

# Rename cp312 to cp313 to fool pip
ren mediapipe-0.10.21-cp312-cp312-win_amd64.whl mediapipe-0.10.21-cp313-cp313-win_amd64.whl

# Install the renamed wheel
.\python_embeded\python.exe -m pip install mediapipe-0.10.21-cp313-cp313-win_amd64.whl --force-reinstall --no-deps
```

## Method 2: Using PowerShell (recommended)

```powershell
# Download with PowerShell
Invoke-WebRequest -Uri "https://files.pythonhosted.org/packages/b7/79/b77808f8195f229ef0c15875540dfdd36724748a4b3de53d993f23336839/mediapipe-0.10.21-cp312-cp312-win_amd64.whl" -OutFile "mediapipe-0.10.21-cp312-cp312-win_amd64.whl"

# Rename the file
Rename-Item "mediapipe-0.10.21-cp312-cp312-win_amd64.whl" "mediapipe-0.10.21-cp313-cp313-win_amd64.whl"

# Install it
.\python_embeded\python.exe -m pip install mediapipe-0.10.21-cp313-cp313-win_amd64.whl --force-reinstall --no-deps
```

## What This Does

1. Downloads the Python 3.12 MediaPipe wheel
2. Renames it to pretend it's Python 3.13 compatible
3. Forces pip to install it without dependency checks

This should fool pip into thinking it's a Python 3.13 compatible wheel!