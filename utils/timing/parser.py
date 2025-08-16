"""
SRT Subtitle Parser for ChatterBox TTS
Handles SRT format parsing with timestamp extraction and validation
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class SRTSubtitle:
    """Represents a single SRT subtitle entry"""
    sequence: int
    start_time: float  # in seconds
    end_time: float    # in seconds
    text: str
    
    @property
    def duration(self) -> float:
        """Duration of the subtitle in seconds"""
        return self.end_time - self.start_time
    
    def __str__(self) -> str:
        return f"SRT({self.sequence}: {self.start_time:.3f}-{self.end_time:.3f}s, '{self.text[:50]}...')"


class SRTParseError(Exception):
    """Exception raised when SRT parsing fails"""
    pass


class SRTParser:
    """
    Parser for SRT subtitle format with comprehensive error handling
    
    SRT Format:
    1
    00:00:01,000 --> 00:00:04,000
    This is the first subtitle
    
    2
    00:00:05,500 --> 00:00:08,200
    This is the second subtitle
    with multiple lines
    """
    
    # Regex pattern for SRT timestamp format: HH:MM:SS,mmm --> HH:MM:SS,mmm
    TIMESTAMP_PATTERN = re.compile(
        r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})'
    )
    
    @staticmethod
    def parse_timestamp(timestamp_str: str) -> float:
        """
        Parse SRT timestamp string to seconds
        
        Args:
            timestamp_str: Timestamp in format "HH:MM:SS,mmm"
            
        Returns:
            Time in seconds as float
            
        Raises:
            SRTParseError: If timestamp format is invalid
        """
        try:
            # Handle both comma and dot as decimal separator
            timestamp_str = timestamp_str.replace(',', '.')
            
            # Split into time components
            time_part, ms_part = timestamp_str.rsplit('.', 1)
            hours, minutes, seconds = map(int, time_part.split(':'))
            milliseconds = int(ms_part)
            
            # Validate ranges
            if not (0 <= hours <= 23):
                raise ValueError(f"Invalid hours: {hours}")
            if not (0 <= minutes <= 59):
                raise ValueError(f"Invalid minutes: {minutes}")
            if not (0 <= seconds <= 59):
                raise ValueError(f"Invalid seconds: {seconds}")
            if not (0 <= milliseconds <= 999):
                raise ValueError(f"Invalid milliseconds: {milliseconds}")
            
            # Convert to total seconds
            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
            return total_seconds
            
        except (ValueError, IndexError) as e:
            raise SRTParseError(f"Invalid timestamp format '{timestamp_str}': {e}")
    
    @staticmethod
    def validate_timing(start_time: float, end_time: float, sequence: int) -> None:
        """
        Validate subtitle timing
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            sequence: Subtitle sequence number for error reporting
            
        Raises:
            SRTParseError: If timing is invalid
        """
        if start_time < 0:
            raise SRTParseError(f"Subtitle {sequence}: Start time cannot be negative ({start_time})")
        
        if end_time < 0:
            raise SRTParseError(f"Subtitle {sequence}: End time cannot be negative ({end_time})")
        
        if start_time >= end_time:
            raise SRTParseError(
                f"Subtitle {sequence}: Start time ({start_time}) must be before end time ({end_time})"
            )
        
        # Check for reasonable duration limits
        duration = end_time - start_time
        if duration > 30.0:  # More than 30 seconds seems excessive
            raise SRTParseError(
                f"Subtitle {sequence}: Duration too long ({duration:.1f}s). Maximum 30s recommended."
            )
        
        if duration < 0.05:  # Less than 50ms seems too short
            raise SRTParseError(
                f"Subtitle {sequence}: Duration too short ({duration:.3f}s). Minimum 0.05s required."
            )
    
    @classmethod
    def parse_srt_content(cls, content: str, allow_overlaps: bool = False) -> List[SRTSubtitle]:
        """
        Parse SRT content string into list of SRTSubtitle objects
        
        Args:
            content: Raw SRT file content as string
            allow_overlaps: If True, allow overlapping subtitles (default: False)
            
        Returns:
            List of SRTSubtitle objects sorted by start time
            
        Raises:
            SRTParseError: If parsing fails
        """
        if not content.strip():
            raise SRTParseError("SRT content is empty")
        
        # Normalize line endings and split into blocks
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        blocks = re.split(r'\n\s*\n', content.strip())
        
        subtitles = []
        
        for block_idx, block in enumerate(blocks):
            if not block.strip():
                continue
                
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            
            if len(lines) < 2: # Sequence and Timing are mandatory
                raise SRTParseError(
                    f"Block {block_idx + 1}: Invalid SRT block format. "
                    f"Expected at least 2 lines (sequence, timing), got {len(lines)}"
                )
            
            try:
                # Parse sequence number
                sequence = int(lines[0])
                if sequence <= 0:
                    raise ValueError(f"Sequence number must be positive, got {sequence}")
                
                # Parse timing line
                timing_line = lines[1]
                match = cls.TIMESTAMP_PATTERN.match(timing_line)
                if not match:
                    raise SRTParseError(
                        f"Block {block_idx + 1}: Invalid timing format '{timing_line}'. "
                        f"Expected format: HH:MM:SS,mmm --> HH:MM:SS,mmm"
                    )
                
                # Extract timestamp components
                start_h, start_m, start_s, start_ms, end_h, end_m, end_s, end_ms = match.groups()
                
                # Convert to seconds
                start_time = cls.parse_timestamp(f"{start_h}:{start_m}:{start_s},{start_ms}")
                end_time = cls.parse_timestamp(f"{end_h}:{end_m}:{end_s},{end_ms}")
                
                # Validate timing
                cls.validate_timing(start_time, end_time, sequence)
                
                # Extract text (everything after timing line)
                # Extract text (everything after timing line)
                if len(lines) >= 3:
                    text_lines = lines[2:]
                    # print(f"🔍 SRT Parser DEBUG: Text lines to join: {text_lines}")
                    # Preserve newlines for character parsing - join with newlines instead of spaces
                    text = '\n'.join(text_lines).strip()
                    # print(f"🔍 SRT Parser DEBUG: Joined text: {repr(text)}")
                else: # len(lines) == 2, implies no text or only whitespace lines that were filtered out
                    text = ""
                
                # Clean up text (remove HTML tags, normalize whitespace but preserve newlines)
                # This will also handle the case where text is already ""
                text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
                # Normalize whitespace but preserve newlines for character parsing
                text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces and tabs only
                text = re.sub(r'\n+', '\n', text)   # Normalize multiple newlines to single
                text = text.strip()  # Remove leading/trailing whitespace
                
                subtitle = SRTSubtitle(
                    sequence=sequence,
                    start_time=start_time,
                    end_time=end_time,
                    text=text
                )
                
                subtitles.append(subtitle)
                
            except ValueError as e:
                raise SRTParseError(f"Block {block_idx + 1}: Invalid sequence number '{lines[0]}': {e}")
            except Exception as e:
                raise SRTParseError(f"Block {block_idx + 1}: Parsing error: {e}")
        
        if not subtitles:
            raise SRTParseError("No valid subtitles found in SRT content")
        
        # Sort by start time and validate sequence
        subtitles.sort(key=lambda s: s.start_time)
        
        # Check for overlapping subtitles (only if not allowing overlaps)
        if not allow_overlaps:
            for i in range(len(subtitles) - 1):
                current = subtitles[i]
                next_sub = subtitles[i + 1]
                
                if current.end_time > next_sub.start_time:
                    raise SRTParseError(
                        f"Overlapping subtitles detected: "
                        f"Subtitle {current.sequence} ends at {current.end_time:.3f}s "
                        f"but subtitle {next_sub.sequence} starts at {next_sub.start_time:.3f}s"
                    )
        
        return subtitles
    
    @classmethod
    def parse_srt_file(cls, file_path: str) -> List[SRTSubtitle]:
        """
        Parse SRT file into list of SRTSubtitle objects
        
        Args:
            file_path: Path to SRT file
            
        Returns:
            List of SRTSubtitle objects sorted by start time
            
        Raises:
            SRTParseError: If file reading or parsing fails
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise SRTParseError(f"SRT file not found: {file_path}")
            
            if not path.is_file():
                raise SRTParseError(f"Path is not a file: {file_path}")
            
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise SRTParseError(f"Could not decode SRT file with any supported encoding: {encodings}")
            
            return cls.parse_srt_content(content)
            
        except Exception as e:
            if isinstance(e, SRTParseError):
                raise
            raise SRTParseError(f"Error reading SRT file '{file_path}': {e}")
    
    @staticmethod
    def get_timing_info(subtitles: List[SRTSubtitle]) -> dict:
        """
        Extract timing information from parsed subtitles
        
        Args:
            subtitles: List of SRTSubtitle objects
            
        Returns:
            Dictionary with timing statistics
        """
        if not subtitles:
            return {
                'total_duration': 0.0,
                'subtitle_count': 0,
                'average_duration': 0.0,
                'gaps': [],
                'total_speech_time': 0.0,
                'total_gap_time': 0.0
            }
        
        total_speech_time = sum(sub.duration for sub in subtitles)
        total_duration = subtitles[-1].end_time
        
        # Calculate gaps between subtitles
        gaps = []
        for i in range(len(subtitles) - 1):
            gap_start = subtitles[i].end_time
            gap_end = subtitles[i + 1].start_time
            if gap_end > gap_start:
                gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration': gap_end - gap_start
                })
        
        total_gap_time = sum(gap['duration'] for gap in gaps)
        
        return {
            'total_duration': total_duration,
            'subtitle_count': len(subtitles),
            'average_duration': total_speech_time / len(subtitles),
            'gaps': gaps,
            'total_speech_time': total_speech_time,
            'total_gap_time': total_gap_time,
            'speech_ratio': total_speech_time / total_duration if total_duration > 0 else 0.0
        }


def validate_srt_timing_compatibility(subtitles: List[SRTSubtitle], 
                                    max_stretch_ratio: float = 2.0,
                                    min_stretch_ratio: float = 0.5) -> List[str]:
    """
    Validate that SRT timing is compatible with TTS generation
    
    Args:
        subtitles: List of parsed SRT subtitles
        max_stretch_ratio: Maximum allowed time stretching ratio
        min_stretch_ratio: Minimum allowed time stretching ratio
        
    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []
    
    for subtitle in subtitles:
        # Estimate natural speech duration (rough estimate: 150 words per minute)
        word_count = len(subtitle.text.split())
        estimated_duration = word_count / 2.5  # 150 words/min = 2.5 words/sec
        
        if estimated_duration > 0:
            stretch_ratio = subtitle.duration / estimated_duration
            
            if stretch_ratio > max_stretch_ratio:
                warnings.append(
                    f"Subtitle {subtitle.sequence}: May require significant time compression "
                    f"(ratio: {stretch_ratio:.2f}x, text: '{subtitle.text[:50]}...')"
                )
            elif stretch_ratio < min_stretch_ratio:
                warnings.append(
                    f"Subtitle {subtitle.sequence}: May require significant time stretching "
                    f"(ratio: {stretch_ratio:.2f}x, text: '{subtitle.text[:50]}...')"
                )
    
    return warnings