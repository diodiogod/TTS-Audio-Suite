"""
Audio Analyzer Node - Interactive waveform visualization and timing extraction
Provides precise word timing extraction for F5TTSEditNode through interactive waveform visualization
"""

import torch
import numpy as np
import os
import tempfile
import json
import time
from typing import Dict, Any, List, Tuple, Optional, Union

# Add project root directory to path for imports
import sys
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio.analysis import AudioAnalyzer, TimingRegion, analysis_cache
from utils.audio.processing import AudioProcessingUtils
import comfy.model_management as model_management


class AudioAnalyzerNode:
    """
    Audio Analyzer Node for interactive waveform visualization and timing extraction.
    Provides precise timing data for F5-TTS speech editing through web interface.
    """
    
    # Enable web interface integration
    WEB_DIRECTORY = "web"
    
    @classmethod
    def NAME(cls):
        return "🌊 Audio Wave Analyzer"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to audio file or drag audio file here.\n\nMouse Controls:\n• Left click + drag: Select audio region\n• Left click on region: Highlight region (green, persistent)\n• Shift + left click: Extend selection\n• Alt + click region: Multi-select for deletion (orange, toggle)\n• Alt + click empty: Clear all multi-selections\n• CTRL + left/right click + drag: Pan waveform\n• Middle mouse + drag: Pan waveform\n• Right click: Clear selection\n• Double click: Seek to position\n• Mouse wheel: Zoom in/out\n• CTRL key: Shows grab cursor for panning\n• Drag amplitude labels (±0.8): Scale waveform vertically\n• Drag loop markers: Move startloop/endloop points\n\nKeyboard Shortcuts:\n• Space: Play/pause\n• Escape: Clear selection\n• Enter: Add selected region\n• Delete: Delete highlighted/selected regions (Shift+Del: clear all)\n• L: Set loop from selection (Shift+L: toggle looping)\n• Shift+C: Clear loop markers\n• Arrow keys: Move playhead (+ Shift for 10s jumps)\n• +/-: Zoom in/out\n• 0: Reset zoom and amplitude scale\n• Home/End: Go to start/end\n\nRegion Management:\n• Click region → highlights green (single, persistent)\n• Alt+click region → selects orange (multiple, toggle)\n• Delete works on both green highlighted and orange selected\n• Regions auto-sort chronologically\n• Manual regions text box: bidirectional sync with interface\n\nLoop Functionality:\n• Select region, then press L or click 'Set Loop'\n• Drag purple loop markers to adjust start/end points\n• Use Shift+L or 'Loop ON/OFF' to enable/disable looping\n• When looping is on, playback repeats between markers\n\nUI Buttons:\n• Upload Audio: Browse and upload audio files\n• Analyze: Process audio with current settings\n• Delete Region: Remove highlighted or selected regions\n• Add Region: Add current selection as new region\n• Clear All: Remove all regions\n• Set Loop: Set loop markers from selection\n• Loop ON/OFF: Toggle loop playback mode\n• Clear Loop: Remove loop markers\n\nNote: Click on the waveform to focus it for keyboard shortcuts",
                    "dynamicPrompts": False
                }),
                "analysis_method": (["silence", "energy", "peaks", "manual"], {
                    "default": "silence",
                    "tooltip": "How to automatically detect speech segments:\n• silence: Finds pauses between words/sentences (best for clear speech)\n• energy: Detects volume changes (good for music or noisy audio)\n• peaks: Finds sharp audio spikes (useful for percussion or effects)\n• manual: Use only manual regions you define below"
                }),
                "precision_level": (["seconds", "milliseconds", "samples"], {
                    "default": "milliseconds",
                    "tooltip": "How precise timing numbers should be in outputs:\n• seconds: Rounded to seconds (1.23s) - for rough timing\n• milliseconds: Precise to milliseconds (1.234s) - for most uses\n• samples: Raw sample numbers (27225 smp) - for exact audio editing"
                }),
                "visualization_points": ("INT", {
                    "default": 2000,
                    "min": 500,
                    "max": 10000,
                    "step": 100,
                    "tooltip": "Waveform detail level - how many points to draw:\n• 500-1000: Smooth waveform, fast rendering\n• 2000-3000: Balanced detail and performance (recommended)\n• 5000-10000: Very detailed, slower but precise for fine editing"
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Connect audio from another node instead of using audio_file path.\nThis input takes priority over the file path if connected."
                }),
                "options": ("ADV_AUDIO_OPTIONS", {
                    "tooltip": "Optional configuration from Audio Analyzer Options node.\nIf connected, uses these advanced settings for analysis.\nIf not connected, uses sensible default values for all analysis methods."
                }),
                "manual_regions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Define your own timing regions manually.\nFormat: start,end (one per line)\nExample:\n1.5,3.2\n4.0,6.8\n8.1,10.5\n\nBidirectional sync:\n• Type/paste here → syncs to interface when you click back\n• Add regions on interface → automatically updates this text\n• Regions auto-sort chronologically by start time\n\nUse when analysis_method is 'manual' or to add extra regions."
                }),
                "region_labels": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional labels for each region (one per line).\nExample:\nIntro\nVerse 1\nChorus\n\nBidirectional sync:\n• Type/paste custom labels here → syncs to interface\n• Interface preserves custom labels when renumbering\n• Auto-generated labels (Region 1, Region 2) get renumbered\n• Custom labels stay unchanged during chronological sorting\n\nMust match the number of manual_regions lines."
                }),
                "export_format": (["f5tts", "json", "csv"], {
                    "default": "f5tts",
                    "tooltip": "How to format the timing_data output:\n• f5tts: Simple format for F5-TTS (start,end per line)\n• json: Full data with confidence, labels, metadata\n• csv: Spreadsheet-compatible format for analysis\n\nAll formats respect the precision_level setting."
                }),
            },
            "hidden": {
                "node_id": ("STRING", {"default": "0"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "AUDIO")
    RETURN_NAMES = ("processed_audio", "timing_data", "analysis_info", "segmented_audio")
    FUNCTION = "analyze_audio"
    CATEGORY = "TTS Audio Suite/🎵 Audio Processing"
    
    def __init__(self):
        self.analyzer = AudioAnalyzer()
        self.temp_files = []
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        self.temp_files.clear()
    
    def __del__(self):
        self.cleanup_temp_files()
    
    def _extract_audio_tensor(self, audio_input: Union[Dict, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Extract audio tensor and sample rate from input.
        Returns BOTH original audio and mono version for analysis.

        Returns:
            Tuple of (original_audio_tensor, mono_audio_for_analysis, sample_rate)
        """
        if isinstance(audio_input, dict):
            if 'waveform' in audio_input:
                audio_tensor = audio_input['waveform']
                sample_rate = audio_input.get('sample_rate', 24000)
            else:
                raise ValueError("Invalid audio format. Expected dictionary with 'waveform' key.")
        elif isinstance(audio_input, torch.Tensor):
            audio_tensor = audio_input
            sample_rate = 24000  # Default sample rate (matches TTS engines)
        else:
            raise ValueError("Invalid audio input type. Expected dict or torch.Tensor.")

        # Keep original for segmentation output (preserves stereo if input was stereo)
        original_audio = audio_tensor

        # Create mono version for analysis (graphs, regions, etc.)
        mono_audio = audio_tensor.clone()

        # Flatten to mono while preserving samples
        while mono_audio.dim() > 1:
            if mono_audio.shape[0] == 1:
                mono_audio = mono_audio.squeeze(0)
            else:
                # Multiple channels: average to mono for analysis
                mono_audio = torch.mean(mono_audio, dim=0)

        if mono_audio.dim() < 1 or mono_audio.dim() > 4:
            raise ValueError(f"Unsupported tensor shape: {audio_tensor.shape}. Expected 1D to 4D tensor.")

        return original_audio, mono_audio, sample_rate
    
    def _parse_manual_regions(self, manual_regions: str, labels: str = "") -> List[TimingRegion]:
        """Parse manual timing regions from multiline string input."""
        if not manual_regions.strip():
            return []
        
        regions = []
        region_lines = [line.strip() for line in manual_regions.strip().split('\n') if line.strip()]
        label_lines = [line.strip() for line in labels.strip().split('\n') if line.strip()] if labels.strip() else []
        
        for i, line in enumerate(region_lines):
            # Handle both comma-separated format (start,end) and semicolon-separated multiple regions
            if ';' in line:
                # Multiple regions in one line (semicolon-separated)
                sub_regions = [r.strip() for r in line.split(';') if r.strip()]
                for j, sub_region in enumerate(sub_regions):
                    if ',' in sub_region:
                        try:
                            start, end = map(float, sub_region.split(','))
                            label = label_lines[i] if i < len(label_lines) else f"region_{len(regions)+1}"
                            
                            regions.append(TimingRegion(
                                start_time=start,
                                end_time=end,
                                label=label,
                                confidence=1.0,
                                metadata={"type": "manual", "source": "user_input"}
                            ))
                        except ValueError:
                            print(f"Warning: Invalid manual region format: '{sub_region}'. Expected 'start,end' format.")
            elif ',' in line:
                # Single region per line
                try:
                    start, end = map(float, line.split(','))
                    label = label_lines[i] if i < len(label_lines) else f"region_{i+1}"
                    
                    regions.append(TimingRegion(
                        start_time=start,
                        end_time=end,
                        label=label,
                        confidence=1.0,
                        metadata={"type": "manual", "source": "user_input"}
                    ))
                except ValueError:
                    print(f"Warning: Invalid manual region format: '{line}'. Expected 'start,end' format.")
        
        return regions
    
    def _format_timing_precision(self, value: float, precision_level: str) -> str:
        """Format timing value according to precision level."""
        if precision_level == "seconds":
            return f"{value:.2f}"
        elif precision_level == "milliseconds":
            return f"{value:.3f}"
        elif precision_level == "samples":
            sample_value = int(value * self.analyzer.sample_rate)
            return str(sample_value)
        else:
            return f"{value:.3f}"
    
    def _get_precision_unit(self, precision_level: str) -> str:
        """Get the unit string for the precision level."""
        if precision_level == "seconds":
            return "s"
        elif precision_level == "milliseconds":
            return "s"
        elif precision_level == "samples":
            return " smp"
        else:
            return "s"
    
    def _apply_precision_to_timing_data(self, timing_data: Any, precision_level: str) -> Any:
        """Apply precision formatting to timing data recursively."""
        if isinstance(timing_data, dict):
            result = {}
            for key, value in timing_data.items():
                if key in ['start', 'end', 'start_time', 'end_time', 'duration'] and isinstance(value, (int, float)):
                    # Format timing values according to precision
                    if precision_level == "samples":
                        result[key] = int(value * self.analyzer.sample_rate)
                    else:
                        result[key] = float(self._format_timing_precision(value, precision_level))
                else:
                    result[key] = self._apply_precision_to_timing_data(value, precision_level)
            return result
        elif isinstance(timing_data, list):
            return [self._apply_precision_to_timing_data(item, precision_level) for item in timing_data]
        else:
            return timing_data
    
    def _format_f5tts_with_precision(self, regions: List[TimingRegion], precision_level: str) -> str:
        """Format timing regions for F5-TTS with precision formatting."""
        if not regions:
            return ""
        
        # Format regions with precision
        formatted_regions = []
        for region in regions:
            start_formatted = self._format_timing_precision(region.start_time, precision_level)
            end_formatted = self._format_timing_precision(region.end_time, precision_level)
            formatted_regions.append(f"{start_formatted},{end_formatted}")
        
        return "\n".join(formatted_regions)
    
    def _create_analysis_info(self, audio_tensor: torch.Tensor, sample_rate: int, 
                             regions: List[TimingRegion], method: str, precision_level: str = "milliseconds",
                             group_threshold: float = 0.0, invert_silence: bool = False, viz_data: Dict = None) -> str:
        """Create analysis information string with precision formatting."""
        duration = AudioProcessingUtils.get_audio_duration(audio_tensor, sample_rate)
        
        # Count grouped vs original regions
        grouped_regions = [r for r in regions if r.metadata and r.metadata.get("type") == "grouped"]
        original_regions = [r for r in regions if not (r.metadata and r.metadata.get("type") == "grouped")]
        
        # Add silence inversion info to method description
        method_description = method
        if method == "silence" and invert_silence:
            method_description = f"{method} (inverted to speech regions)"
        
        info_lines = [
            f"Audio Analysis Results",
            f"Duration: {self._format_timing_precision(duration, precision_level)} {self._get_precision_unit(precision_level)}",
            f"Sample Rate: {sample_rate} Hz",
            f"Analysis Method: {method_description}",
            f"Regions Found: {len(regions)}",
        ]
        
        # Add grouping information if applicable
        if group_threshold > 0.0:
            total_original = sum(r.metadata.get("source_regions", 1) for r in grouped_regions) + len(original_regions)
            info_lines.extend([
                f"",
                f"Region Grouping:",
                f"  Grouping Threshold: {group_threshold:.3f}s",
                f"  Original Regions: {total_original}",
                f"  Final Regions: {len(regions)} ({len(grouped_regions)} grouped, {len(original_regions)} individual)",
                f"  Regions Merged: {total_original - len(regions)}"
            ])
        
        info_lines.extend([
            f"",
            "Timing Regions:"
        ])
        
        for i, region in enumerate(regions):
            region_duration = region.end_time - region.start_time
            start_formatted = self._format_timing_precision(region.start_time, precision_level)
            end_formatted = self._format_timing_precision(region.end_time, precision_level)
            duration_formatted = self._format_timing_precision(region_duration, precision_level)
            unit = self._get_precision_unit(precision_level)
            
            # Add grouping details for grouped regions
            grouping_info = ""
            if region.metadata and region.metadata.get("type") == "grouped":
                source_count = region.metadata.get("source_regions", 0)
                original_labels = region.metadata.get("original_labels", [])
                grouping_info = f" [grouped from {source_count} regions: {', '.join(original_labels)}]"
            
            info_lines.append(
                f"  {i+1}. {region.label}: {start_formatted}{unit} - {end_formatted}{unit} "
                f"(duration: {duration_formatted}{unit}, confidence: {region.confidence:.2f}){grouping_info}"
            )
        
        # Add visualization summary if available
        if viz_data:
            info_lines.extend([
                f"",
                "Visualization Summary:",
                f"  Waveform Points: {len(viz_data.get('samples', []))}",
                f"  Duration: {viz_data.get('duration', 0):.3f}s",
                f"  Sample Rate: {viz_data.get('sample_rate', 0)} Hz"
            ])
            
            if viz_data.get('rms'):
                rms_data = viz_data['rms']
                if isinstance(rms_data, dict) and 'values' in rms_data:
                    info_lines.append(f"  RMS Data Points: {len(rms_data['values'])}")
                elif isinstance(rms_data, list):
                    info_lines.append(f"  RMS Data Points: {len(rms_data)}")
        
        return "\n".join(info_lines)
    
    def _create_segmented_audio(self, audio_tensor: torch.Tensor, sample_rate: int, regions: List[TimingRegion]) -> torch.Tensor:
        """
        Create audio containing only the detected regions, concatenated together.
        
        Args:
            audio_tensor: Original audio tensor
            sample_rate: Sample rate of the audio
            regions: List of TimingRegion objects to extract
            
        Returns:
            New audio tensor with only the selected regions
        """
        if not regions:
            # No regions detected, return short silence in ComfyUI format
            silence = torch.zeros(1, int(0.1 * sample_rate))  # 0.1 second of silence
            return AudioProcessingUtils.format_for_comfyui(silence, sample_rate)
        
        # Sort regions by start time
        sorted_regions = sorted(regions, key=lambda r: r.start_time)
        
        segments = []
        for region in sorted_regions:
            # Convert time to sample indices
            start_sample = int(region.start_time * sample_rate)
            end_sample = int(region.end_time * sample_rate)
            
            # Ensure indices are within bounds
            audio_length = audio_tensor.shape[-1] if audio_tensor.dim() > 1 else len(audio_tensor)
            start_sample = max(0, min(start_sample, audio_length))
            end_sample = max(start_sample, min(end_sample, audio_length))
            
            if end_sample > start_sample:
                # Extract the audio segment
                if audio_tensor.dim() == 1:
                    segment = audio_tensor[start_sample:end_sample]
                else:
                    segment = audio_tensor[:, start_sample:end_sample]
                segments.append(segment)
        
        if not segments:
            # No valid segments found, return silence in ComfyUI format
            silence = torch.zeros(1, int(0.1 * sample_rate))
            return AudioProcessingUtils.format_for_comfyui(silence, sample_rate)
        
        # Concatenate all segments
        if audio_tensor.dim() == 1:
            segmented_audio = torch.cat(segments, dim=0)
        else:
            segmented_audio = torch.cat(segments, dim=1)
        
        # Format for ComfyUI
        return AudioProcessingUtils.format_for_comfyui(segmented_audio, sample_rate)
    
    def analyze_audio(self, audio_file, analysis_method="silence", precision_level="milliseconds",
                     visualization_points=2000, audio=None, options=None, manual_regions="", region_labels="", 
                     export_format="f5tts", node_id=""):
        """
        Analyze audio for timing extraction and visualization.
        
        Args:
            audio: Input audio data
            analysis_method: Method for timing detection
            precision_level: Precision level for output
            visualization_points: Number of points for visualization
            silence_threshold: Threshold for silence detection
            silence_min_duration: Minimum duration for silence regions
            energy_sensitivity: Sensitivity for energy-based detection
            manual_regions: Manual timing regions string
            region_labels: Labels for regions
            export_format: Export format for timing data
            
        Returns:
            Tuple of (processed_audio, timing_data, analysis_info, segmented_audio)
        """
        
        try:
            # Set up default values for technical parameters
            # These are sensible defaults that work well for most use cases
            silence_threshold = 0.01
            silence_min_duration = 0.1
            invert_silence_regions = False
            energy_sensitivity = 0.5
            peak_threshold = 0.02
            peak_min_distance = 0.05
            peak_region_size = 0.1
            group_regions_threshold = 0.000
            
            # Handle options input - if provided, use options values over defaults
            if options is not None and isinstance(options, dict):
                # Extract technical parameters from options
                silence_threshold = options.get("silence_threshold", silence_threshold)
                silence_min_duration = options.get("silence_min_duration", silence_min_duration)
                invert_silence_regions = options.get("invert_silence_regions", invert_silence_regions)
                energy_sensitivity = options.get("energy_sensitivity", energy_sensitivity)
                peak_threshold = options.get("peak_threshold", peak_threshold)
                peak_min_distance = options.get("peak_min_distance", peak_min_distance)
                peak_region_size = options.get("peak_region_size", peak_region_size)
                group_regions_threshold = options.get("group_regions_threshold", group_regions_threshold)
            
            # Handle audio input - either from file or from input
            if audio is not None:
                # Audio input from another node
                original_audio, mono_audio, sample_rate = self._extract_audio_tensor(audio)
            elif audio_file and audio_file.strip():
                # Load audio from file path
                file_path = audio_file.strip()

                # If path is not absolute, try to resolve it relative to ComfyUI input directory
                if not os.path.isabs(file_path):
                    try:
                        import folder_paths
                        input_dir = folder_paths.get_input_directory()
                        full_path = os.path.join(input_dir, file_path)
                        if os.path.exists(full_path):
                            file_path = full_path
                            # print(f"🎵 Resolved relative path to: {file_path}")  # Debug: path resolution
                    except ImportError:
                        print("⚠️ Could not import folder_paths, using path as-is")

                if not os.path.exists(file_path):
                    print(f"❌ Audio file not found: {file_path}")
                    raise FileNotFoundError(f"Audio file not found: {file_path}")
                audio_tensor, sample_rate = self.analyzer.load_audio(file_path)
                # For file input, create mono version for analysis
                original_audio = audio_tensor
                mono_audio = audio_tensor.clone()
                while mono_audio.dim() > 1:
                    if mono_audio.shape[0] == 1:
                        mono_audio = mono_audio.squeeze(0)
                    else:
                        mono_audio = torch.mean(mono_audio, dim=0)
            else:
                raise ValueError("No audio input provided. Either connect an audio input or specify an audio file path.")

            # Set analyzer sample rate
            self.analyzer.sample_rate = sample_rate
            
            # Generate cache key for analysis
            # Use mono tensor for cache key (consistent across stereo/mono)
            tensor_hash = hash((tuple(mono_audio.shape), float(mono_audio.mean()), float(mono_audio.std())))
            manual_hash = hash((manual_regions, region_labels))  # Include manual regions in cache
            cache_key = f"{tensor_hash}_{analysis_method}_{silence_threshold}_{silence_min_duration}_{invert_silence_regions}_{energy_sensitivity}_{peak_threshold}_{peak_min_distance}_{peak_region_size}_{group_regions_threshold}_{manual_hash}"

            # Check cache first
            cached_result = analysis_cache.get(cache_key)
            if cached_result:
                regions = cached_result
                # print("📋 Using cached analysis results")  # Debug: cache usage
            else:
                # Perform analysis on MONO version only
                if analysis_method == "manual":
                    regions = self._parse_manual_regions(manual_regions, region_labels)
                elif analysis_method == "silence":
                    regions = self.analyzer.detect_silence_regions(
                        mono_audio, threshold=silence_threshold, min_duration=silence_min_duration,
                        invert=invert_silence_regions
                    )
                elif analysis_method == "energy":
                    regions = self.analyzer.detect_word_boundaries(
                        mono_audio, sensitivity=energy_sensitivity
                    )
                elif analysis_method == "peaks":
                    regions = self.analyzer.extract_timing_regions(
                        mono_audio, method="peaks",
                        peak_threshold=peak_threshold,
                        peak_min_distance=peak_min_distance,
                        peak_region_size=peak_region_size
                    )
                else:
                    raise ValueError(f"Unknown analysis method: {analysis_method}")
                
                # Add manual regions to auto-detected regions (if any manual regions exist)
                manual_regions_list = self._parse_manual_regions(manual_regions, region_labels)
                if manual_regions_list and analysis_method != "manual":
                    # Combine auto-detected and manual regions
                    regions.extend(manual_regions_list)
                    # Sort all regions by start time
                    regions.sort(key=lambda r: r.start_time)
                
                # Apply region grouping if threshold > 0
                if group_regions_threshold > 0.000:
                    regions = self.analyzer.group_regions(regions, group_regions_threshold)
                
                # Cache results
                analysis_cache.put(cache_key, regions)
            
            # Generate visualization data from MONO version
            viz_data = self.analyzer.generate_visualization_data(mono_audio, visualization_points)

            # Add regions to visualization data
            viz_data["regions"] = [
                {
                    "start": float(region.start_time),
                    "end": float(region.end_time),
                    "label": str(region.label),
                    "confidence": float(region.confidence),
                    "metadata": region.metadata or {}
                }
                for region in regions
            ]

            # Format timing data according to export format and precision level
            if export_format == "f5tts":
                # F5TTS format with precision formatting
                timing_data = self._format_f5tts_with_precision(regions, precision_level)
            else:
                # Apply precision formatting to exported timing data
                raw_timing_data = self.analyzer.export_timing_data(regions, export_format)
                formatted_timing_data = self._apply_precision_to_timing_data(raw_timing_data, precision_level)
                timing_data = json.dumps(formatted_timing_data, indent=2)

            # Create analysis info from MONO version
            analysis_info = self._create_analysis_info(mono_audio, sample_rate, regions, analysis_method, precision_level, group_regions_threshold, invert_silence_regions, viz_data)

            # Return ORIGINAL audio in ComfyUI format (passthrough - preserves stereo if input was stereo)
            processed_audio = AudioProcessingUtils.format_for_comfyui(original_audio, sample_rate)
            
            # Generate segmented audio from ORIGINAL (preserves stereo if input was stereo)
            segmented_audio = self._create_segmented_audio(original_audio, sample_rate, regions)
            
            
            # Save visualization data to ComfyUI temp directory and save audio for web access
            try:
                import folder_paths
                import shutil
                import soundfile as sf
                
                # Save visualization data
                temp_dir = folder_paths.get_temp_directory()
                temp_file = os.path.join(temp_dir, f"audio_data_{node_id}.json")
                
                # Add audio file path to visualization data for JavaScript
                web_audio_filename = None
                
                # Handle audio file copying or saving - respect priority: connected audio first
                if audio is not None:
                    # Connected audio: save ORIGINAL tensor to temporary file for web access (preserves stereo)
                    try:
                        input_dir = folder_paths.get_input_directory()
                        # Use unique filename per analysis so JavaScript detects file change
                        # Include tensor hash to invalidate cache when audio input changes
                        import hashlib
                        audio_hash = hashlib.md5(str(tensor_hash).encode()).hexdigest()[:8]
                        temp_audio_filename = f"connected_audio_{node_id}_{audio_hash}.wav"
                        temp_audio_path = os.path.join(input_dir, temp_audio_filename)

                        # Convert original tensor to numpy array for soundfile (preserves stereo/channels)
                        audio_numpy = original_audio.cpu().numpy()
                        if audio_numpy.ndim == 1:
                            # Mono audio
                            sf.write(temp_audio_path, audio_numpy, sample_rate)
                        elif audio_numpy.ndim == 2:
                            # Stereo/multichannel - save as-is
                            if audio_numpy.shape[0] == 1:
                                # Single channel in 2D format
                                sf.write(temp_audio_path, audio_numpy[0], sample_rate)
                            else:
                                # Multiple channels - transpose for soundfile [channels, samples] -> [samples, channels]
                                sf.write(temp_audio_path, audio_numpy.T, sample_rate)
                        else:
                            # 3D or 4D - flatten to 2D first
                            while audio_numpy.ndim > 2:
                                audio_numpy = audio_numpy.squeeze(0)
                            if audio_numpy.shape[0] == 1:
                                sf.write(temp_audio_path, audio_numpy[0], sample_rate)
                            else:
                                sf.write(temp_audio_path, audio_numpy.T, sample_rate)
                        
                        web_audio_filename = temp_audio_filename
                        # print(f"🎵 Connected audio saved for web access: {temp_audio_path}")  # Debug: audio save
                        
                    except Exception as audio_save_error:
                        print(f"⚠️ Failed to save connected audio: {audio_save_error}")  # Keep: important error
                        # Continue without audio playback for connected audio
                
                elif audio_file and audio_file.strip() and os.path.exists(audio_file.strip()):
                    # File-based audio: copy to ComfyUI input directory for web access
                    input_dir = folder_paths.get_input_directory()
                    audio_filename = os.path.basename(audio_file.strip())
                    web_audio_path = os.path.join(input_dir, audio_filename)
                    
                    # Copy if not already there or if source is newer
                    if not os.path.exists(web_audio_path) or os.path.getmtime(audio_file.strip()) > os.path.getmtime(web_audio_path):
                        shutil.copy2(audio_file.strip(), web_audio_path)
                        # print(f"🎵 Audio file copied for web access: {web_audio_path}")  # Debug: file copy
                    
                    # For file-based audio, provide just the filename for web access
                    # JavaScript will use this with ComfyUI's input URL format
                    file_path_for_js = audio_filename
                
                # Add audio information to visualization data for JavaScript
                if web_audio_filename:
                    # Connected audio - provide web_audio_filename
                    viz_data["web_audio_filename"] = web_audio_filename
                elif 'file_path_for_js' in locals():
                    # File-based audio - provide file_path
                    viz_data["file_path"] = file_path_for_js
                
                with open(temp_file, 'w') as f:
                    json.dump(viz_data, f, indent=2)
                
                # print(f"🎵 Audio data saved to temp: {temp_file}")  # Debug: temp file save
                
            except Exception as save_error:
                print(f"⚠️ Audio Analyzer data save failed: {save_error}")  # Keep: important error
                # Continue without failing the entire analysis
            
            return (processed_audio, timing_data, analysis_info, segmented_audio)
            
        except Exception as e:
            import traceback
            error_msg = f"Audio analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"Full traceback: {traceback.format_exc()}")
            
            # Return error data
            empty_audio = torch.zeros(1, 1000)  # 1 second of silence
            processed_audio = AudioProcessingUtils.format_for_comfyui(empty_audio, 24000)
            segmented_audio = processed_audio  # Same as processed for errors
            
            return (
                processed_audio,
                f"Error: {error_msg}",
                f"Analysis failed: {error_msg}",
                segmented_audio
            )
    
    def validate_inputs(self, **inputs) -> Dict[str, Any]:
        """Validate node inputs."""
        validated = {}
        
        # Validate required inputs - audio can be from file or input
        if "audio" not in inputs and "audio_file" not in inputs:
            raise ValueError("Either audio input or audio_file is required")
        
        validated["audio"] = inputs["audio"]
        validated["options"] = inputs.get("options", None)
        validated["analysis_method"] = inputs.get("analysis_method", "silence")
        validated["precision_level"] = inputs.get("precision_level", "milliseconds")
        validated["visualization_points"] = max(500, min(10000, inputs.get("visualization_points", 2000)))
        
        # Validate optional inputs (only main parameters, technical parameters come from options node)
        validated["manual_regions"] = inputs.get("manual_regions", "")
        validated["region_labels"] = inputs.get("region_labels", "")
        validated["export_format"] = inputs.get("export_format", "f5tts")
        
        return validated


# Node class mappings for registration
NODE_CLASS_MAPPINGS = {
    "AudioAnalyzerNode": AudioAnalyzerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioAnalyzerNode": "🌊 Audio Wave Analyzer"
}