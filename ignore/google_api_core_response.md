# Discord Response: Google API Core Issue

**Why `google.api_core` is needed:**

`google.api_core` is a low-level library used by `google-cloud-storage`, which F5-TTS uses to download models from cloud storage. It's just a technical dependency - not a Google surveillance component.

**You're right to question this though** - if models are local, you shouldn't need cloud storage access. Unfortunately, the current F5-TTS integration has a flaw where it still tries to access cloud storage even when models exist locally.

**Quick fix:**

```bash
pip install google-api-core google-cloud-storage
```

**If you really don't want Google packages:**

1. **Use different TTS engines** - ChatterBox and Higgs Audio don't need Google packages
2. **Try force reinstalling** if the auto-installer failed:
   ```bash
   pip install --force-reinstall google-api-core google-cloud-storage cached-path
   ```

**Technical explanation:**
F5-TTS uses `cached-path` library → which uses `google-cloud-storage` → which needs `google.api_core`. It's just for accessing cloud storage where models are hosted, similar to downloading from GitHub or Hugging Face. The issue is our integration doesn't always pass local model paths properly, so it falls back to cloud downloads.

**Good news!** This has been fixed in the latest version - F5-TTS now properly uses local models without needing Google packages when models are downloaded manually.

**Bottom line:** This is a legitimate technical dependency (not Google spyware), but you're right that it shouldn't be needed for local models. For now, ChatterBox engine works great and has zero Google dependencies.