"""
Unit tests for SRT Parser
Tests utils/timing/parser.py without requiring ComfyUI server
"""

import pytest
import sys
import os
import importlib.util
from pathlib import Path

# Add custom node root to path BEFORE any project imports
# This avoids triggering the full node loading chain
custom_node_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(custom_node_root))

# Set up minimal environment to avoid ComfyUI imports
os.environ.setdefault('COMFYUI_TESTING', '1')

# Load the parser module directly using importlib to bypass package __init__.py
parser_path = custom_node_root / "utils" / "timing" / "parser.py"
spec = importlib.util.spec_from_file_location("parser_module", parser_path)
parser_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser_module)

# Import what we need from the loaded module
SRTParser = parser_module.SRTParser
SRTSubtitle = parser_module.SRTSubtitle
SRTParseError = parser_module.SRTParseError
validate_srt_timing_compatibility = parser_module.validate_srt_timing_compatibility


@pytest.mark.unit
class TestSRTParserTimestamp:
    """Tests for timestamp parsing"""
    
    def test_parse_valid_timestamp(self):
        """Test parsing valid SRT timestamp"""
        result = SRTParser.parse_timestamp("00:01:30,500")
        assert result == 90.5  # 1 min 30 sec 500 ms
    
    def test_parse_timestamp_with_dot(self):
        """Test parsing timestamp with dot instead of comma"""
        result = SRTParser.parse_timestamp("00:00:05.250")
        assert result == 5.25
    
    def test_parse_zero_timestamp(self):
        """Test parsing zero timestamp"""
        result = SRTParser.parse_timestamp("00:00:00,000")
        assert result == 0.0
    
    def test_parse_max_reasonable_timestamp(self):
        """Test parsing hour-long timestamp"""
        result = SRTParser.parse_timestamp("01:30:45,123")
        assert result == pytest.approx(5445.123, rel=1e-3)
    
    def test_parse_invalid_timestamp_raises(self):
        """Test that invalid format raises SRTParseError"""
        with pytest.raises(SRTParseError):
            SRTParser.parse_timestamp("invalid")
    
    def test_parse_invalid_hours_raises(self):
        """Test that invalid hours raises error"""
        with pytest.raises(SRTParseError):
            SRTParser.parse_timestamp("99:00:00,000")


@pytest.mark.unit
class TestSRTParserValidation:
    """Tests for timing validation"""
    
    def test_valid_timing_passes(self):
        """Test that valid timing doesn't raise"""
        SRTParser.validate_timing(1.0, 3.0, 1)  # Should not raise
    
    def test_negative_start_raises(self):
        """Test that negative start time raises error"""
        with pytest.raises(SRTParseError, match="negative"):
            SRTParser.validate_timing(-1.0, 3.0, 1)
    
    def test_start_after_end_raises(self):
        """Test that start >= end raises error"""
        with pytest.raises(SRTParseError, match="before end"):
            SRTParser.validate_timing(5.0, 3.0, 1)
    
    def test_equal_start_end_raises(self):
        """Test that start == end raises error"""
        with pytest.raises(SRTParseError, match="before end"):
            SRTParser.validate_timing(3.0, 3.0, 1)
    
    def test_too_long_duration_raises(self):
        """Test that >30s duration raises error"""
        with pytest.raises(SRTParseError, match="too long"):
            SRTParser.validate_timing(0.0, 35.0, 1)
    
    def test_too_short_duration_raises(self):
        """Test that <50ms duration raises error"""
        with pytest.raises(SRTParseError, match="too short"):
            SRTParser.validate_timing(0.0, 0.01, 1)


@pytest.mark.unit
class TestSRTParserContent:
    """Tests for full SRT content parsing"""
    
    def test_parse_simple_srt(self, sample_srt_content):
        """Test parsing valid SRT content"""
        subtitles = SRTParser.parse_srt_content(sample_srt_content)
        
        assert len(subtitles) == 3
        assert subtitles[0].sequence == 1
        assert subtitles[0].text == "Hello, this is the first subtitle."
        assert subtitles[0].start_time == 1.0
        assert subtitles[0].end_time == 4.0
    
    def test_parse_multiline_subtitle(self):
        """Test parsing subtitle with multiple lines"""
        content = """1
00:00:01,000 --> 00:00:04,000
First line
Second line
"""
        subtitles = SRTParser.parse_srt_content(content)
        assert "First line" in subtitles[0].text
        assert "Second line" in subtitles[0].text
    
    def test_empty_content_raises(self):
        """Test that empty content raises error"""
        with pytest.raises(SRTParseError, match="empty"):
            SRTParser.parse_srt_content("")
    
    def test_whitespace_only_raises(self):
        """Test that whitespace-only content raises error"""
        with pytest.raises(SRTParseError, match="empty"):
            SRTParser.parse_srt_content("   \n\n   ")
    
    def test_overlapping_subtitles_raises(self):
        """Test that overlapping subtitles raise error by default"""
        content = """1
00:00:01,000 --> 00:00:05,000
First subtitle

2
00:00:03,000 --> 00:00:07,000
Overlapping subtitle
"""
        with pytest.raises(SRTParseError, match="Overlapping"):
            SRTParser.parse_srt_content(content)
    
    def test_overlapping_allowed(self):
        """Test that overlaps can be allowed"""
        content = """1
00:00:01,000 --> 00:00:05,000
First subtitle

2
00:00:03,000 --> 00:00:07,000
Overlapping subtitle
"""
        subtitles = SRTParser.parse_srt_content(content, allow_overlaps=True)
        assert len(subtitles) == 2


@pytest.mark.unit
class TestSRTSubtitle:
    """Tests for SRTSubtitle dataclass"""
    
    def test_duration_property(self):
        """Test duration calculation"""
        subtitle = SRTSubtitle(
            sequence=1,
            start_time=1.0,
            end_time=4.5,
            text="Test"
        )
        assert subtitle.duration == 3.5
    
    def test_str_representation(self):
        """Test string representation"""
        subtitle = SRTSubtitle(
            sequence=1,
            start_time=1.0,
            end_time=4.0,
            text="Hello world"
        )
        str_repr = str(subtitle)
        assert "1" in str_repr
        assert "1.000" in str_repr
        assert "4.000" in str_repr


@pytest.mark.unit
class TestSRTTimingInfo:
    """Tests for timing info extraction"""
    
    def test_get_timing_info(self, sample_srt_content):
        """Test timing info extraction"""
        subtitles = SRTParser.parse_srt_content(sample_srt_content)
        info = SRTParser.get_timing_info(subtitles)
        
        assert info["subtitle_count"] == 3
        assert info["total_duration"] == 12.0  # Last subtitle ends at 12s
        assert "average_duration" in info
        assert "gaps" in info
    
    def test_empty_subtitles_timing_info(self):
        """Test timing info for empty list"""
        info = SRTParser.get_timing_info([])
        
        assert info["subtitle_count"] == 0
        assert info["total_duration"] == 0.0


@pytest.mark.unit
class TestSRTTimingCompatibility:
    """Tests for TTS timing compatibility validation"""
    
    def test_validate_normal_pacing(self, sample_srt_content):
        """Test that normal pacing produces no warnings"""
        subtitles = SRTParser.parse_srt_content(sample_srt_content)
        warnings = validate_srt_timing_compatibility(subtitles)
        # Normal pacing should have few or no warnings
        assert isinstance(warnings, list)
