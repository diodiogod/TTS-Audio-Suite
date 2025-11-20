"""
Voice Fixer Model Downloader
Downloads required VoiceFixer models to TTS/ folder structure using UnifiedDownloader
"""

import os
from utils.downloads.unified_downloader import unified_downloader
from utils.models.extra_paths import get_preferred_download_path


class VoiceFixerDownloader:
    """
    Manages VoiceFixer model downloads to organized TTS/voicefixer/ structure
    Uses UnifiedDownloader for consistency with other engines
    """

    # HuggingFace repository (mirrored from official Zenodo release for faster downloads)
    HF_REPO = "Diogodiogod/voicefixer-models"

    # Model filenames
    ANALYSIS_MODULE_FILE = "vf.ckpt"
    VOCODER_FILE = "model.ckpt-1490000_trimed.pt"

    def __init__(self):
        self.downloader = unified_downloader
        # Uses get_preferred_download_path with engine_name to respect extra_model_paths.yaml
        self.voicefixer_dir = get_preferred_download_path('TTS', engine_name='voicefixer')

    def get_analysis_module_path(self) -> str:
        """Get path to VoiceFixer analysis module checkpoint"""
        return os.path.join(self.voicefixer_dir, self.ANALYSIS_MODULE_FILE)

    def get_vocoder_path(self) -> str:
        """Get path to VoiceFixer vocoder checkpoint (44.1kHz)"""
        return os.path.join(self.voicefixer_dir, self.VOCODER_FILE)

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

        print("📥 VoiceFixer models required - downloading from HuggingFace...")
        print(f"   Repository: {self.HF_REPO}")
        print(f"   Download location: {self.voicefixer_dir}")

        # Download analysis module if missing using UnifiedDownloader
        if not os.path.exists(analysis_path):
            success = self.downloader.download_from_hf_cli(
                self.HF_REPO,
                self.ANALYSIS_MODULE_FILE,
                self.voicefixer_dir
            )
            if not success:
                print(f"\n❌ Failed to download analysis module")
                print(f"   Manual download: https://huggingface.co/{self.HF_REPO}")
                print(f"   Save to: {analysis_path}")
                return False

        # Download vocoder if missing using UnifiedDownloader
        if not os.path.exists(vocoder_path):
            success = self.downloader.download_from_hf_cli(
                self.HF_REPO,
                self.VOCODER_FILE,
                self.voicefixer_dir
            )
            if not success:
                print(f"\n❌ Failed to download vocoder")
                print(f"   Manual download: https://huggingface.co/{self.HF_REPO}")
                print(f"   Save to: {vocoder_path}")
                return False

        return True
