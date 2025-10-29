"""
Voice Fixer Model Downloader
Downloads required VoiceFixer models to TTS/ folder structure with ComfyUI model management integration
"""

import os
from typing import Optional
import requests
from utils.downloads.unified_downloader import UnifiedDownloader
from utils.models.extra_paths import get_preferred_download_path


class VoiceFixerDownloader:
    """
    Manages VoiceFixer model downloads to organized TTS/voicefixer/ structure
    Uses UnifiedDownloader for consistent model management across the suite
    """

    # Model URLs from Zenodo (official VoiceFixer releases)
    ANALYSIS_MODULE_URL = "https://zenodo.org/record/5600188/files/vf.ckpt?download=1"
    VOCODER_URL = "https://zenodo.org/record/5600188/files/model.ckpt-1490000_trimed.pt?download=1"

    def __init__(self):
        self.downloader = UnifiedDownloader()
        # Uses get_preferred_download_path with engine_name to respect extra_model_paths.yaml
        self.voicefixer_dir = get_preferred_download_path('TTS', engine_name='voicefixer')

    def get_analysis_module_path(self) -> str:
        """Get path to VoiceFixer analysis module checkpoint"""
        return os.path.join(self.voicefixer_dir, 'vf.ckpt')

    def get_vocoder_path(self) -> str:
        """Get path to VoiceFixer vocoder checkpoint (44.1kHz)"""
        return os.path.join(self.voicefixer_dir, 'model.ckpt-1490000_trimed.pt')

    def _download_with_resume(self, url: str, target_path: str, description: str, max_retries: int = 3) -> bool:
        """
        Download file with resume capability and retries.

        Args:
            url: Download URL
            target_path: Where to save the file
            description: Human-readable description
            max_retries: Number of retry attempts

        Returns:
            True if successful, False otherwise
        """
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        temp_path = target_path + '.tmp'

        for attempt in range(max_retries):
            try:
                # Start fresh if previous attempt was incomplete
                if attempt > 0 and os.path.exists(temp_path):
                    os.remove(temp_path)

                # Download with streaming
                print(f"📥 Downloading {description}...")
                response = requests.get(url, stream=True, timeout=60, verify=False)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0

                # Write to temp file
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                print(f"\r📥 {os.path.basename(target_path)}: {progress:.1f}%", end='', flush=True)

                print()  # New line after progress

                # Verify download completed
                temp_size = os.path.getsize(temp_path)
                if total_size > 0 and temp_size < total_size:
                    raise RuntimeError(f"Incomplete download: got {temp_size} bytes, expected {total_size}")

                # Move temp file to final location
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.rename(temp_path, target_path)

                print(f"✅ Downloaded: {target_path}")
                return True

            except Exception as e:
                print(f"\n⚠️ Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"   Retrying in 5 seconds... (attempt {attempt + 2}/{max_retries})")
                    import time
                    time.sleep(5)

        return False

    def ensure_models_downloaded(self) -> bool:
        """
        Ensure both required VoiceFixer models are downloaded
        Returns True if successful, False if download failed
        """
        analysis_path = self.get_analysis_module_path()
        vocoder_path = self.get_vocoder_path()

        # Check if models already exist
        models_exist = os.path.exists(analysis_path) and os.path.exists(vocoder_path)
        if models_exist:
            return True

        print("📥 VoiceFixer models required - downloading from Zenodo...")
        print(f"   Download location: {self.voicefixer_dir}")

        # Download analysis module if missing
        if not os.path.exists(analysis_path):
            success = self._download_with_resume(
                self.ANALYSIS_MODULE_URL,
                analysis_path,
                "VoiceFixer analysis module (vf.ckpt)",
                max_retries=3
            )
            if not success:
                print(f"\n❌ Failed to download analysis module after 3 attempts")
                print(f"   Manual download: {self.ANALYSIS_MODULE_URL}")
                print(f"   Save to: {analysis_path}")
                return False

        # Download vocoder if missing
        if not os.path.exists(vocoder_path):
            success = self._download_with_resume(
                self.VOCODER_URL,
                vocoder_path,
                "VoiceFixer 44.1kHz vocoder",
                max_retries=3
            )
            if not success:
                print(f"\n❌ Failed to download vocoder after 3 attempts")
                print(f"   Manual download: {self.VOCODER_URL}")
                print(f"   Save to: {vocoder_path}")
                return False

        return True
