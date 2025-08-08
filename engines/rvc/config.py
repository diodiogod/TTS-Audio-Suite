"""
RVC Configuration - Core configuration for RVC inference
Based on reference implementation
"""

import torch
import os


class RVCConfig:
    def __init__(self):
        # Device configuration
        self.device = self._get_optimal_device()
        self.is_half = self.device != "cpu" and torch.cuda.is_available()
        
        # Audio configuration
        self.t_pad = 320 * 8  # Padding for audio processing
        self.t_pad_tgt = 320 * 8
        self.window = 160
        
        # Model configuration
        self.hubert_model = None
        
        # Cache directories
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "rvc")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_optimal_device(self):
        """Get optimal device for inference"""
        if torch.cuda.is_available():
            return f"cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def __str__(self):
        return f"RVCConfig(device={self.device}, is_half={self.is_half})"


# Global config instance
config = RVCConfig()