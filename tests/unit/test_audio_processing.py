"""
Unit tests for Audio Processing Utilities
Tests utils/audio/processing.py without requiring ComfyUI server
"""

import pytest
import sys
import os
import torch
from pathlib import Path

# Add custom node root to path BEFORE any project imports
custom_node_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(custom_node_root))

# Set up minimal environment to avoid ComfyUI imports
os.environ.setdefault('COMFYUI_TESTING', '1')

from utils.audio.processing import AudioProcessingUtils


@pytest.mark.unit
class TestTimeConversions:
    """Tests for time/sample conversions"""
    
    def test_seconds_to_samples(self):
        """Test seconds to samples conversion"""
        samples = AudioProcessingUtils.seconds_to_samples(1.0, 22050)
        assert samples == 22050
    
    def test_seconds_to_samples_fractional(self):
        """Test fractional seconds conversion"""
        samples = AudioProcessingUtils.seconds_to_samples(0.5, 44100)
        assert samples == 22050
    
    def test_samples_to_seconds(self):
        """Test samples to seconds conversion"""
        seconds = AudioProcessingUtils.samples_to_seconds(22050, 22050)
        assert seconds == 1.0
    
    def test_samples_to_seconds_fractional(self):
        """Test fractional samples conversion"""
        seconds = AudioProcessingUtils.samples_to_seconds(11025, 22050)
        assert seconds == 0.5
    
    def test_conversion_roundtrip(self):
        """Test roundtrip conversion"""
        original = 2.5
        samples = AudioProcessingUtils.seconds_to_samples(original, 22050)
        result = AudioProcessingUtils.samples_to_seconds(samples, 22050)
        assert result == pytest.approx(original, rel=1e-6)


@pytest.mark.unit
class TestSilenceCreation:
    """Tests for silence tensor creation"""
    
    def test_create_silence_basic(self):
        """Test basic silence creation"""
        silence = AudioProcessingUtils.create_silence(
            duration_seconds=1.0,
            sample_rate=22050
        )
        
        # Mono silence is 1D tensor
        assert silence.shape == (22050,)
        assert torch.all(silence == 0)
        assert silence.dtype == torch.float32
    
    def test_create_silence_stereo(self):
        """Test stereo silence creation"""
        silence = AudioProcessingUtils.create_silence(
            duration_seconds=0.5,
            sample_rate=44100,
            channels=2
        )
        
        assert silence.shape == (2, 22050)
    
    def test_create_silence_custom_dtype(self):
        """Test silence with custom dtype"""
        silence = AudioProcessingUtils.create_silence(
            duration_seconds=0.1,
            sample_rate=22050,
            dtype=torch.float16
        )
        
        assert silence.dtype == torch.float16


@pytest.mark.unit
class TestAudioNormalization:
    """Tests for audio tensor normalization"""
    
    def test_normalize_1d_tensor(self):
        """Test that 1D tensor stays 1D (mono audio)"""
        audio = torch.randn(22050)
        normalized = AudioProcessingUtils.normalize_audio_tensor(audio)
        
        # API keeps 1D audio as 1D
        assert normalized.dim() == 1
        assert normalized.shape == (22050,)
    
    def test_normalize_2d_tensor_unchanged(self):
        """Test that 2D tensor with correct shape is unchanged"""
        audio = torch.randn(1, 22050)
        normalized = AudioProcessingUtils.normalize_audio_tensor(audio)
        
        assert normalized.shape == audio.shape
    
    def test_normalize_3d_tensor(self):
        """Test normalizing 3D tensor"""
        audio = torch.randn(1, 1, 22050)
        normalized = AudioProcessingUtils.normalize_audio_tensor(audio)
        
        assert normalized.dim() == 2


@pytest.mark.unit
class TestAudioDuration:
    """Tests for audio duration calculation"""
    
    def test_get_duration_1d(self):
        """Test duration calculation for 1D tensor"""
        audio = torch.randn(22050)
        duration = AudioProcessingUtils.get_audio_duration(audio, 22050)
        
        assert duration == pytest.approx(1.0, rel=1e-6)
    
    def test_get_duration_2d(self):
        """Test duration calculation for 2D tensor"""
        audio = torch.randn(1, 44100)
        duration = AudioProcessingUtils.get_audio_duration(audio, 22050)
        
        assert duration == pytest.approx(2.0, rel=1e-6)


@pytest.mark.unit
class TestAudioConcatenation:
    """Tests for audio segment concatenation"""
    
    def test_concatenate_simple(self):
        """Test simple concatenation"""
        seg1 = torch.randn(1, 1000)
        seg2 = torch.randn(1, 500)
        
        result = AudioProcessingUtils.concatenate_audio_segments(
            [seg1, seg2],
            method="simple"
        )
        
        assert result.shape == (1, 1500)
    
    def test_concatenate_with_silence(self):
        """Test concatenation with silence gap"""
        seg1 = torch.randn(1, 1000)
        seg2 = torch.randn(1, 1000)
        
        result = AudioProcessingUtils.concatenate_audio_segments(
            [seg1, seg2],
            method="silence",
            silence_duration=0.1,
            sample_rate=10000  # 0.1s = 1000 samples
        )
        
        # 1000 + 1000 + 1000 (silence) = 3000
        assert result.shape == (1, 3000)
    
    def test_concatenate_empty_list(self):
        """Test concatenating empty list"""
        result = AudioProcessingUtils.concatenate_audio_segments([])
        
        assert result.shape[-1] == 0


@pytest.mark.unit
class TestCrossfade:
    """Tests for audio crossfading"""
    
    def test_crossfade_basic(self):
        """Test basic crossfade between two segments"""
        seg1 = torch.ones(1, 2000)
        seg2 = torch.ones(1, 2000) * 0.5
        
        result = AudioProcessingUtils.crossfade_audio(
            seg1, seg2,
            fade_duration=0.1,
            sample_rate=10000  # 0.1s = 1000 samples
        )
        
        # Total should be less than sum (overlap)
        assert result.shape[-1] < 4000
        assert result.shape[-1] == 3000  # 2000 + 2000 - 1000
    
    def test_crossfade_short_audio(self):
        """Test crossfade with audio shorter than fade"""
        seg1 = torch.ones(1, 500)
        seg2 = torch.ones(1, 500)
        
        # Should handle gracefully
        result = AudioProcessingUtils.crossfade_audio(
            seg1, seg2,
            fade_duration=1.0,  # Longer than audio
            sample_rate=1000
        )
        
        assert result.shape[-1] > 0


@pytest.mark.unit
class TestPadding:
    """Tests for audio padding"""
    
    def test_pad_to_duration_end(self):
        """Test padding at end"""
        # Use 2D audio since padding uses 2D silence
        audio = torch.randn(1, 11025)  # 0.5s at 22050, 2D format
        
        padded = AudioProcessingUtils.pad_audio_to_duration(
            audio,
            target_duration=1.0,
            sample_rate=22050,
            pad_type="end"
        )
        
        assert padded.shape == (1, 22050)
        # Original content preserved at start
        assert torch.allclose(padded[:, :11025], audio)
    
    def test_pad_to_duration_start(self):
        """Test padding at start"""
        # Use 2D audio since padding uses 2D silence  
        audio = torch.randn(1, 11025)
        
        padded = AudioProcessingUtils.pad_audio_to_duration(
            audio,
            target_duration=1.0,
            sample_rate=22050,
            pad_type="start"
        )
        
        assert padded.shape == (1, 22050)
        # Original content preserved at end
        assert torch.allclose(padded[:, -11025:], audio)
    
    def test_pad_already_long_enough(self):
        """Test padding audio that's already long enough"""
        audio = torch.randn(1, 44100)  # 2s
        
        padded = AudioProcessingUtils.pad_audio_to_duration(
            audio,
            target_duration=1.0,  # Shorter than audio
            sample_rate=22050
        )
        
        # Should return original (no truncation)
        assert padded.shape == audio.shape


@pytest.mark.unit
class TestComfyUIFormat:
    """Tests for ComfyUI format conversion"""
    
    def test_format_for_comfyui(self):
        """Test formatting for ComfyUI output"""
        audio = torch.randn(1, 22050)
        
        result = AudioProcessingUtils.format_for_comfyui(audio, 22050)
        
        assert isinstance(result, dict)
        assert "waveform" in result
        assert "sample_rate" in result
        assert result["sample_rate"] == 22050
