"""
SRT Reporting - Handles report generation and SRT output formatting
Extracted from the massive SRT TTS node for better maintainability
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


class SRTReportGenerator:
    """
    Handles generation of timing reports and adjusted SRT content
    """
    
    def __init__(self):
        """Initialize report generator"""
        pass
    
    def generate_timing_report(self, subtitles: List, adjustments: List[Dict], timing_mode: str) -> str:
        """Generate detailed timing report - EXACT ORIGINAL LOGIC FROM LINES 1291-1431"""
        import numpy as np
        
        # Handle edge case where no subtitles were processed (immediate interruption)
        if not subtitles:
            return f"""SRT Timing Report ({timing_mode} mode)
{'=' * 50}
Total subtitles: 0 (interrupted immediately)
Total duration: 0.000s

No segments were processed due to immediate interruption.
"""
        
        report_lines = [
            f"SRT Timing Report ({timing_mode} mode)",
            "=" * 50,
            f"Total subtitles: {len(subtitles)}",
            f"Total duration: {subtitles[-1].end_time:.3f}s",
            "",
            "Per-subtitle analysis:"
        ]
        
        if timing_mode == "smart_natural":
            # For smart_natural mode, iterate directly over the detailed adjustments report
            for adj in adjustments:
                report_lines.append(
                    f"  {adj['sequence']:2d}. Original SRT: {adj['original_srt_start']:6.2f}-{adj['original_srt_end']:6.2f}s "
                    f"(Target: {adj['original_srt_duration']:.2f}s)"
                )
                report_lines.append(f"      Natural Audio: {adj['natural_audio_duration']:.3f}s")
                
                for action in adj['actions']:
                    report_lines.append(f"      - {action}")
                
                report_lines.append(
                    f"      Final Audio Duration: {adj['final_segment_duration']:.3f}s "
                    f"(Final SRT: {adj['final_srt_start']:6.2f}-{adj['final_srt_end']:6.2f}s)"
                )
                # Find the corresponding subtitle to get its text
                # This assumes subtitles are sorted by sequence or index
                original_subtitle_text = next((s.text for s in subtitles if s.sequence == adj['sequence']), "N/A")
                report_lines.append(f"      Text: {original_subtitle_text[:60]}{'...' if len(original_subtitle_text) > 60 else ''}")
        else:
            # For other modes, iterate using zip with original subtitles
            for i, (subtitle, adj) in enumerate(zip(subtitles, adjustments)):
                if timing_mode == "pad_with_silence":
                    # For pad_with_silence mode, show overlap/gap information
                    timing_info = ""
                    if adj['natural_duration'] > subtitle.duration:
                        # Audio is longer than SRT slot - will overlap
                        overlap = adj['natural_duration'] - subtitle.duration
                        timing_info = f" 🔁 [OVERLAP: +{overlap:.2f}s]"
                    elif i < len(subtitles) - 1:
                        # Check for silence gap to next subtitle
                        next_subtitle = subtitles[i + 1]
                        gap_duration = next_subtitle.start_time - subtitle.end_time
                        if gap_duration > 0:
                            timing_info = f" [+{gap_duration:.2f}s silence]"
                    
                    report_lines.append(
                        f"  {subtitle.sequence:2d}. {subtitle.start_time:6.2f}-{subtitle.end_time:6.2f}s "
                        f"({subtitle.duration:.2f}s target, {adj['natural_duration']:.2f}s natural){timing_info}"
                    )
                else:
                    # For other modes (e.g., stretch_to_fit), show stretch information
                    stretch_info = ""
                    if adj['needs_stretching']:
                        stretch_info = f" [{adj['stretch_type']} {adj['stretch_factor']:.2f}x]"
                    
                    report_lines.append(
                        f"  {subtitle.sequence:2d}. {subtitle.start_time:6.2f}-{subtitle.end_time:6.2f}s "
                        f"({subtitle.duration:.2f}s target, {adj['natural_duration']:.2f}s natural){stretch_info}"
                    )
                
                report_lines.append(f"      Text: {subtitle.text[:60]}{'...' if len(subtitle.text) > 60 else ''}")
        
        # Summary statistics
        if timing_mode == "pad_with_silence":
            total_gaps = 0
            total_gap_duration = 0
            total_overlaps = 0
            total_overlap_duration = 0
            
            for i, (subtitle, adj) in enumerate(zip(subtitles, adjustments)):
                if adj['natural_duration'] > subtitle.duration:
                    total_overlaps += 1
                    total_overlap_duration += adj['natural_duration'] - subtitle.duration
                elif i < len(subtitles) - 1:
                    gap_duration = subtitles[i + 1].start_time - subtitles[i].end_time
                    if gap_duration > 0:
                        total_gaps += 1
                        total_gap_duration += gap_duration
            
            summary_lines = [
                "",
                "Summary:",
                f"  Audio preserved at natural timing (no stretching)"
            ]
            
            if total_overlaps > 0:
                summary_lines.append(f"  Timing overlaps: {total_overlaps} segments, +{total_overlap_duration:.2f}s total overlap")
            
            if total_gaps > 0:
                summary_lines.append(f"  Silence gaps added: {total_gaps} gaps, {total_gap_duration:.2f}s total silence")
            
            if total_overlaps == 0 and total_gaps == 0:
                summary_lines.append(f"  Perfect timing match - no gaps or overlaps")
            
            report_lines.extend(summary_lines)
        elif timing_mode == "smart_natural":
            # Define the same threshold used in smart timing processing
            INSIGNIFICANT_TRUNCATION_THRESHOLD = 0.05
            
            total_shifted = sum(1 for adj in adjustments if adj['next_segment_shifted_by'] > 0)
            total_stretched = sum(1 for adj in adjustments if abs(adj['stretch_factor_applied'] - 1.0) > 0.01)
            total_padded = sum(1 for adj in adjustments if adj['padding_added'] > 0)
            # Only count significant truncations in the summary
            total_truncated = sum(1 for adj in adjustments if adj['truncated_by'] > INSIGNIFICANT_TRUNCATION_THRESHOLD)
            
            summary_lines = [
                "",
                "Summary (Smart Natural Mode):",
                f"  Segments with next segment shifted: {total_shifted}/{len(adjustments)}",
                f"  Segments with audio stretched/shrunk: {total_stretched}/{len(adjustments)}",
                f"  Segments with silence padded: {total_padded}/{len(adjustments)}",
                f"  Segments truncated: {total_truncated}/{len(adjustments)}",
            ]
            report_lines.extend(summary_lines)
        else:
            total_stretch_needed = sum(1 for adj in adjustments if adj['needs_stretching'])
            avg_stretch = np.mean([adj['stretch_factor'] for adj in adjustments])
            
            report_lines.extend([
                "",
                "Summary:",
                f"  Segments needing time adjustment: {total_stretch_needed}/{len(adjustments)}",
                f"  Average stretch factor: {avg_stretch:.2f}x",
            ])
        
        return "\n".join(report_lines)
    
    def generate_adjusted_srt_string(self, subtitles: List, adjustments: List[Dict], timing_mode: str) -> str:
        """Generate adjusted SRT string from final timings - EXACT ORIGINAL LOGIC"""
        if not adjustments:
            return "# No subtitles processed\n"
        
        srt_lines = []
        for i, adj in enumerate(adjustments):
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                seconds_int = int(seconds % 60)
                milliseconds = int((seconds - int(seconds)) * 1000)
                return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"
            
            # Determine timing based on mode
            if timing_mode == "smart_natural":
                start_time = adj.get('final_srt_start', adj.get('original_srt_start', 0))
                end_time = start_time + adj.get('final_segment_duration', adj.get('natural_audio_duration', 1))
            else:
                start_time = adj.get('start_time', subtitles[i].start_time if i < len(subtitles) else 0)
                end_time = adj.get('end_time', subtitles[i].end_time if i < len(subtitles) else 1)
            
            start_time_str = format_time(start_time)
            end_time_str = format_time(end_time)
            
            # Get text
            text = adj.get('original_text', subtitles[i].text if i < len(subtitles) else f"Subtitle {i+1}")
            
            srt_lines.append(str(adj.get('sequence', i+1)))
            srt_lines.append(f"{start_time_str} --> {end_time_str}")
            srt_lines.append(text)
            srt_lines.append("")
        
        return "\n".join(srt_lines)
    
    def generate_generation_info(self, total_duration: float, segment_count: int, 
                               timing_mode: str, cache_status: str, 
                               model_source: str) -> str:
        """
        Generate concise generation info string
        
        Args:
            total_duration: Total audio duration in seconds
            segment_count: Number of segments processed
            timing_mode: Timing mode used
            cache_status: Cache status ("cached", "generated", "mixed")
            model_source: Model source information
            
        Returns:
            Formatted generation info string
        """
        return (f"Generated {total_duration:.1f}s SRT-timed audio from {segment_count} subtitles "
                f"using {timing_mode} mode ({cache_status} segments, {model_source} models)")
    
    def _generate_smart_natural_analysis(self, adjustments: List[Dict]) -> List[str]:
        """Generate analysis for smart natural timing mode"""
        analysis_lines = []
        
        for adj in adjustments:
            sequence = adj.get('sequence', adj.get('segment_index', 0) + 1)
            original_start = adj.get('original_srt_start', 0)
            original_end = adj.get('original_srt_end', 0)
            original_duration = adj.get('original_srt_duration', original_end - original_start)
            
            analysis_lines.append(
                f"  {sequence:2d}. Original SRT: {original_start:6.2f}-{original_end:6.2f}s "
                f"(Target: {original_duration:.2f}s)"
            )
            
            # Add actions taken
            actions = adj.get('actions', ['No actions recorded'])
            for action in actions:
                analysis_lines.append(f"      - {action}")
            
            # Add timing deviation info if available
            if 'timing_deviation' in adj:
                deviation = adj['timing_deviation']
                within_tolerance = adj.get('within_tolerance', True)
                status = "✓" if within_tolerance else "⚠"
                analysis_lines.append(f"      {status} Timing deviation: {deviation:.3f}s")
            
            # Add text preview
            text = adj.get('original_text', 'No text available')
            preview = text[:60] + '...' if len(text) > 60 else text
            analysis_lines.append(f"      Text: {preview}")
            analysis_lines.append("")  # Empty line between entries
        
        return analysis_lines
    
    def _generate_standard_analysis(self, subtitles: List, adjustments: List[Dict]) -> List[str]:
        """Generate analysis for standard timing modes"""
        analysis_lines = []
        
        for i, adj in enumerate(adjustments):
            if i < len(subtitles):
                subtitle = subtitles[i]
                sequence = subtitle.sequence
                start_time = subtitle.start_time
                end_time = subtitle.end_time
                duration = subtitle.duration
                text = subtitle.text
            else:
                sequence = adj.get('sequence', i + 1)
                start_time = adj.get('start_time', 0)
                end_time = adj.get('end_time', 1)
                duration = end_time - start_time
                text = adj.get('original_text', f'Subtitle {i+1}')
            
            # Format stretch information
            stretch_info = ""
            if adj.get('needs_stretching'):
                stretch_type = adj.get('stretch_type', 'unknown')
                stretch_factor = adj.get('stretch_factor', 1.0)
                stretch_info = f" [{stretch_type} {stretch_factor:.2f}x]"
            
            natural_duration = adj.get('natural_duration', duration)
            
            analysis_lines.append(
                f"  {sequence:2d}. {start_time:6.2f}-{end_time:6.2f}s "
                f"({duration:.2f}s target, {natural_duration:.2f}s natural){stretch_info}"
            )
            
            # Add text preview
            preview = text[:60] + '...' if len(text) > 60 else text
            analysis_lines.append(f"      Text: {preview}")
        
        return analysis_lines
    
    def _generate_summary_statistics(self, adjustments: List[Dict], timing_mode: str) -> List[str]:
        """Generate summary statistics from adjustments"""
        stats_lines = []
        
        # Count different types of adjustments
        stretched_count = sum(1 for adj in adjustments if adj.get('needs_stretching', False))
        no_change_count = len(adjustments) - stretched_count
        
        if timing_mode == "smart_natural":
            # Smart natural specific stats
            shifted_count = sum(1 for adj in adjustments if adj.get('next_segment_shifted_by', 0) > 0)
            padded_count = sum(1 for adj in adjustments if adj.get('padding_added', 0) > 0)
            
            stats_lines.extend([
                f"- Segments using natural timing: {no_change_count}",
                f"- Segments requiring adjustment: {stretched_count}",
                f"- Segments causing shifts: {shifted_count}",
                f"- Segments with padding: {padded_count}"
            ])
            
            # Calculate average timing deviation
            deviations = [adj.get('timing_deviation', 0) for adj in adjustments if 'timing_deviation' in adj]
            if deviations:
                avg_deviation = sum(deviations) / len(deviations)
                max_deviation = max(deviations)
                stats_lines.append(f"- Average timing deviation: {avg_deviation:.3f}s")
                stats_lines.append(f"- Maximum timing deviation: {max_deviation:.3f}s")
        
        else:
            # Standard timing mode stats
            if stretched_count > 0:
                compress_count = sum(1 for adj in adjustments 
                                   if adj.get('stretch_type') == 'compress')
                expand_count = sum(1 for adj in adjustments 
                                 if adj.get('stretch_type') == 'expand')
                
                stats_lines.extend([
                    f"- Segments requiring stretching: {stretched_count}",
                    f"  - Compressed: {compress_count}",
                    f"  - Extended: {expand_count}",
                    f"- Segments with natural timing: {no_change_count}"
                ])
            else:
                stats_lines.append(f"- All {len(adjustments)} segments used natural timing")
        
        return stats_lines
    
    def _format_srt_timestamp(self, seconds: float) -> str:
        """
        Format seconds as SRT timestamp (HH:MM:SS,mmm)
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_int = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"
    
    def generate_debug_report(self, subtitles: List, adjustments: List[Dict], 
                            audio_segments: List = None) -> str:
        """
        Generate detailed debug report for troubleshooting
        
        Args:
            subtitles: List of SRTSubtitle objects
            adjustments: List of timing adjustments
            audio_segments: Optional list of audio segments for analysis
            
        Returns:
            Detailed debug report string
        """
        debug_lines = [
            "SRT Processing Debug Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Input subtitles: {len(subtitles) if subtitles else 0}",
            f"Generated adjustments: {len(adjustments)}",
            f"Audio segments: {len(audio_segments) if audio_segments else 'N/A'}",
            ""
        ]
        
        # Add detailed adjustment analysis
        if adjustments:
            debug_lines.append("Detailed Adjustment Analysis:")
            for i, adj in enumerate(adjustments):
                debug_lines.append(f"Adjustment {i+1}:")
                for key, value in adj.items():
                    if isinstance(value, float):
                        debug_lines.append(f"  {key}: {value:.6f}")
                    elif isinstance(value, list):
                        debug_lines.append(f"  {key}: {len(value)} items")
                    else:
                        debug_lines.append(f"  {key}: {value}")
                debug_lines.append("")
        
        return "\n".join(debug_lines)