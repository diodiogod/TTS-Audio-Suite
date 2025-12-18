"""
Unit tests for Pause Tag Processor
Tests utils/text/pause_processor.py without requiring ComfyUI server
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

from utils.text.pause_processor import PauseTagProcessor


@pytest.mark.unit
class TestPauseTagDetection:
    """Tests for pause tag detection"""
    
    def test_has_pause_tags_positive(self):
        """Test detection of pause tags"""
        assert PauseTagProcessor.has_pause_tags("Hello [pause:2] world")
        assert PauseTagProcessor.has_pause_tags("Test [wait:1.5] text")
        assert PauseTagProcessor.has_pause_tags("Check [stop:500ms] this")
    
    def test_has_pause_tags_negative(self):
        """Test no false positives"""
        assert not PauseTagProcessor.has_pause_tags("Hello world")
        assert not PauseTagProcessor.has_pause_tags("[other:2] tag")
        assert not PauseTagProcessor.has_pause_tags("No tags here")
    
    def test_has_pause_tags_case_variations(self):
        """Test case-insensitive detection"""
        assert PauseTagProcessor.has_pause_tags("[Pause:2]")
        assert PauseTagProcessor.has_pause_tags("[PAUSE:2]")
        assert PauseTagProcessor.has_pause_tags("[Wait:1]")
        assert PauseTagProcessor.has_pause_tags("[STOP:1]")


@pytest.mark.unit
class TestPauseTagParsing:
    """Tests for pause tag parsing"""
    
    def test_parse_simple_pause(self):
        """Test parsing simple pause tag"""
        segments, clean = PauseTagProcessor.parse_pause_tags("Hello [pause:2] world")
        
        assert len(segments) == 3
        assert segments[0] == ("text", "Hello")
        assert segments[1] == ("pause", 2.0)
        assert segments[2] == ("text", "world")
        assert clean == "Hello world"
    
    def test_parse_milliseconds(self):
        """Test parsing millisecond pause"""
        segments, clean = PauseTagProcessor.parse_pause_tags("Test [pause:500ms] more")
        
        pause_segment = [s for s in segments if s[0] == "pause"][0]
        assert pause_segment[1] == 0.5  # 500ms = 0.5s
    
    def test_parse_explicit_seconds(self):
        """Test parsing explicit seconds"""
        segments, clean = PauseTagProcessor.parse_pause_tags("Test [pause:2s] more")
        
        pause_segment = [s for s in segments if s[0] == "pause"][0]
        assert pause_segment[1] == 2.0
    
    def test_parse_decimal_duration(self):
        """Test parsing decimal duration"""
        segments, clean = PauseTagProcessor.parse_pause_tags("Test [pause:1.5] more")
        
        pause_segment = [s for s in segments if s[0] == "pause"][0]
        assert pause_segment[1] == 1.5
    
    def test_parse_multiple_pauses(self, sample_text_with_pauses):
        """Test parsing multiple pause tags"""
        segments, clean = PauseTagProcessor.parse_pause_tags(sample_text_with_pauses)
        
        pause_segments = [s for s in segments if s[0] == "pause"]
        assert len(pause_segments) == 2
        assert pause_segments[0][1] == 2.0  # [pause:2]
        assert pause_segments[1][1] == 0.5  # [wait:500ms]
    
    def test_parse_wait_alias(self):
        """Test wait alias for pause"""
        segments, _ = PauseTagProcessor.parse_pause_tags("Test [wait:3] here")
        
        pause_segment = [s for s in segments if s[0] == "pause"][0]
        assert pause_segment[1] == 3.0
    
    def test_parse_stop_alias(self):
        """Test stop alias for pause"""
        segments, _ = PauseTagProcessor.parse_pause_tags("Test [stop:1.5] here")
        
        pause_segment = [s for s in segments if s[0] == "pause"][0]
        assert pause_segment[1] == 1.5
    
    def test_no_pause_tags(self):
        """Test text without pause tags"""
        segments, clean = PauseTagProcessor.parse_pause_tags("No pauses here")
        
        assert len(segments) == 1
        assert segments[0] == ("text", "No pauses here")
        assert clean == "No pauses here"
    
    def test_clean_text_whitespace(self):
        """Test clean text has normalized whitespace"""
        segments, clean = PauseTagProcessor.parse_pause_tags(
            "Hello   [pause:2]    world"
        )
        assert clean == "Hello world"


@pytest.mark.unit
class TestSilenceGeneration:
    """Tests for silence segment generation"""
    
    def test_create_silence_basic(self):
        """Test basic silence creation"""
        silence = PauseTagProcessor.create_silence_segment(
            duration_seconds=1.0,
            sample_rate=22050
        )
        
        assert silence.shape == (1, 22050)
        assert torch.all(silence == 0)
    
    def test_create_silence_custom_rate(self):
        """Test silence with custom sample rate"""
        silence = PauseTagProcessor.create_silence_segment(
            duration_seconds=0.5,
            sample_rate=44100
        )
        
        assert silence.shape == (1, 22050)  # 0.5 * 44100
    
    def test_create_silence_clamped_max(self):
        """Test silence clamped to max 30 seconds"""
        silence = PauseTagProcessor.create_silence_segment(
            duration_seconds=60.0,  # Request 60s
            sample_rate=1000
        )
        
        # Should be clamped to 30s
        assert silence.shape == (1, 30000)
    
    def test_create_silence_clamped_min(self):
        """Test silence clamped to min 0 seconds"""
        silence = PauseTagProcessor.create_silence_segment(
            duration_seconds=-5.0,  # Negative
            sample_rate=1000
        )
        
        assert silence.shape == (1, 0)


@pytest.mark.unit
class TestPreprocess:
    """Tests for preprocessing convenience function"""
    
    def test_preprocess_with_tags_enabled(self):
        """Test preprocessing with pause tags enabled"""
        text, segments = PauseTagProcessor.preprocess_text_with_pause_tags(
            "Hello [pause:2] world",
            enable_pause_tags=True
        )
        
        assert text == "Hello world"
        assert segments is not None
        assert len(segments) == 3
    
    def test_preprocess_with_tags_disabled(self):
        """Test preprocessing with pause tags disabled"""
        original = "Hello [pause:2] world"
        text, segments = PauseTagProcessor.preprocess_text_with_pause_tags(
            original,
            enable_pause_tags=False
        )
        
        assert text == original
        assert segments is None
    
    def test_preprocess_no_tags_present(self):
        """Test preprocessing when no tags present"""
        original = "Hello world"
        text, segments = PauseTagProcessor.preprocess_text_with_pause_tags(
            original,
            enable_pause_tags=True
        )
        
        assert text == original
        assert segments is None
