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

# Add parent directory to path for imports
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.audio_analysis import AudioAnalyzer, TimingRegion, analysis_cache
from core.audio_processing import AudioProcessingUtils
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
        return "🎵 Audio Analyzer"
    
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
                "options": ("OPTIONS", {
                    "tooltip": "Optional configuration from Audio Analyzer Options node.\nIf connected, these settings override the individual parameter widgets below.\nIf not connected, uses the individual parameter values or defaults."
                }),
                "silence_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "How quiet audio must be to count as silence (0.001-0.1):\n• 0.001-0.005: Very sensitive, catches whispers as speech\n• 0.01: Default, good for most recordings\n• 0.05-0.1: Less sensitive, ignores background noise\nOnly used when analysis_method is 'silence'"
                }),
                "silence_min_duration": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Shortest pause to count as a break between words (0.01-2.0 seconds):\n• 0.01-0.05: Catches tiny pauses between syllables\n• 0.1: Default, good for word breaks\n• 0.5-2.0: Only long pauses between sentences\nOnly used when analysis_method is 'silence'"
                }),
                "energy_sensitivity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "How sensitive to detect volume changes (0.0-1.0):\n• 0.0-0.3: Very sensitive, detects small volume changes\n• 0.5: Default, balanced detection\n• 0.7-1.0: Less sensitive, only major volume changes\nOnly used when analysis_method is 'energy'"
                }),
                "peak_threshold": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.001,
                    "max": 0.5,
                    "step": 0.001,
                    "tooltip": "Minimum amplitude threshold for detecting peaks (0.001-0.5):\n• 0.001-0.01: Very sensitive, catches soft consonants and emphasis\n• 0.02: Default for speech, good for normal speaking volume\n• 0.05-0.1: Less sensitive, only strong emphasis or loud sounds\n• 0.2-0.5: Only very loud peaks\nOnly used when analysis_method is 'peaks'"
                }),
                "peak_min_distance": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum time between detected peaks in seconds (0.01-1.0):\n• 0.01-0.03: Very sensitive, catches rapid syllables\n• 0.05: Default for speech, good for normal speech pace\n• 0.1-0.2: Less sensitive, only distinct words/emphasis\n• 0.5-1.0: Only major speech events\nOnly used when analysis_method is 'peaks'"
                }),
                "peak_region_size": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.02,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Size of timing region around each peak in seconds (0.02-1.0):\n• 0.02-0.05: Tight regions for precise timing\n• 0.1: Default, good balance for speech editing\n• 0.2-0.5: Wider regions for context around peaks\n• 0.5-1.0: Very wide regions for phrase-level editing\nOnly used when analysis_method is 'peaks'"
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
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "AUDIO")
    RETURN_NAMES = ("timing_data", "visualization_data", "analysis_info", "processed_audio")
    FUNCTION = "analyze_audio"
    CATEGORY = "ChatterBox Audio"
    
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
    
    def _extract_audio_tensor(self, audio_input: Union[Dict, torch.Tensor]) -> Tuple[torch.Tensor, int]:
        """Extract audio tensor and sample rate from input."""
        if isinstance(audio_input, dict):
            if 'waveform' in audio_input:
                audio_tensor = audio_input['waveform']
                sample_rate = audio_input.get('sample_rate', 22050)
            else:
                raise ValueError("Invalid audio format. Expected dictionary with 'waveform' key.")
        elif isinstance(audio_input, torch.Tensor):
            audio_tensor = audio_input
            sample_rate = 22050  # Default sample rate
        else:
            raise ValueError("Invalid audio input type. Expected dict or torch.Tensor.")
        
        # Normalize audio tensor
        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)  # Remove batch dimension
        
        if audio_tensor.dim() == 2 and audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)  # Convert to mono
        
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.squeeze(0)  # Remove channel dimension if mono
        
        return audio_tensor, sample_rate
    
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
                             regions: List[TimingRegion], method: str, precision_level: str = "milliseconds") -> str:
        """Create analysis information string with precision formatting."""
        duration = AudioProcessingUtils.get_audio_duration(audio_tensor, sample_rate)
        
        info_lines = [
            f"Audio Analysis Results",
            f"Duration: {self._format_timing_precision(duration, precision_level)} {self._get_precision_unit(precision_level)}",
            f"Sample Rate: {sample_rate} Hz",
            f"Analysis Method: {method}",
            f"Regions Found: {len(regions)}",
            f"",
            "Timing Regions:"
        ]
        
        for i, region in enumerate(regions):
            region_duration = region.end_time - region.start_time
            start_formatted = self._format_timing_precision(region.start_time, precision_level)
            end_formatted = self._format_timing_precision(region.end_time, precision_level)
            duration_formatted = self._format_timing_precision(region_duration, precision_level)
            unit = self._get_precision_unit(precision_level)
            
            info_lines.append(
                f"  {i+1}. {region.label}: {start_formatted}{unit} - {end_formatted}{unit} "
                f"(duration: {duration_formatted}{unit}, confidence: {region.confidence:.2f})"
            )
        
        return "\n".join(info_lines)
    
    def analyze_audio(self, audio_file, analysis_method="silence", precision_level="milliseconds",
                     visualization_points=2000, audio=None, options=None, silence_threshold=0.01, silence_min_duration=0.1,
                     energy_sensitivity=0.5, peak_threshold=0.02, peak_min_distance=0.05, peak_region_size=0.1,
                     manual_regions="", region_labels="", export_format="f5tts", node_id=""):
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
            Tuple of (timing_data, visualization_data, analysis_info, processed_audio)
        """
        
        try:
            # Handle options input - if provided, use options values over individual parameters
            if options is not None and isinstance(options, dict):
                # Extract values from options, falling back to current parameter values if not in options
                analysis_method = options.get("analysis_method", analysis_method)
                silence_threshold = options.get("silence_threshold", silence_threshold)
                silence_min_duration = options.get("silence_min_duration", silence_min_duration)
                energy_sensitivity = options.get("energy_sensitivity", energy_sensitivity)
                peak_threshold = options.get("peak_threshold", peak_threshold)
                peak_min_distance = options.get("peak_min_distance", peak_min_distance)
                peak_region_size = options.get("peak_region_size", peak_region_size)
                manual_regions = options.get("manual_regions", manual_regions)
                region_labels = options.get("region_labels", region_labels)
                export_format = options.get("export_format", export_format)
            
            # Handle audio input - either from file or from input
            if audio is not None:
                # Audio input from another node
                audio_tensor, sample_rate = self._extract_audio_tensor(audio)
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
            else:
                raise ValueError("No audio input provided. Either connect an audio input or specify an audio file path.")
            
            # Set analyzer sample rate
            self.analyzer.sample_rate = sample_rate
            
            # Generate cache key for analysis
            # Use tensor shape and mean for more stable caching
            tensor_hash = hash((tuple(audio_tensor.shape), float(audio_tensor.mean()), float(audio_tensor.std())))
            cache_key = f"{tensor_hash}_{analysis_method}_{silence_threshold}_{silence_min_duration}_{energy_sensitivity}_{peak_threshold}_{peak_min_distance}_{peak_region_size}"
            
            # Check cache first
            cached_result = analysis_cache.get(cache_key)
            if cached_result:
                regions = cached_result
                # print("📋 Using cached analysis results")  # Debug: cache usage
            else:
                # Perform analysis based on method
                if analysis_method == "manual":
                    regions = self._parse_manual_regions(manual_regions, region_labels)
                elif analysis_method == "silence":
                    regions = self.analyzer.detect_silence_regions(
                        audio_tensor, threshold=silence_threshold, min_duration=silence_min_duration
                    )
                elif analysis_method == "energy":
                    regions = self.analyzer.detect_word_boundaries(
                        audio_tensor, sensitivity=energy_sensitivity
                    )
                elif analysis_method == "peaks":
                    regions = self.analyzer.extract_timing_regions(
                        audio_tensor, method="peaks", 
                        peak_threshold=peak_threshold, 
                        peak_min_distance=peak_min_distance,
                        peak_region_size=peak_region_size
                    )
                else:
                    raise ValueError(f"Unknown analysis method: {analysis_method}")
                
                # Cache results
                analysis_cache.put(cache_key, regions)
            
            # Generate visualization data
            viz_data = self.analyzer.generate_visualization_data(audio_tensor, visualization_points)
            
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
            
            # Create analysis info with precision formatting
            analysis_info = self._create_analysis_info(audio_tensor, sample_rate, regions, analysis_method, precision_level)
            
            # Format visualization data as JSON
            visualization_json = json.dumps(viz_data, indent=2)
            
            # Return processed audio in ComfyUI format
            processed_audio = AudioProcessingUtils.format_for_comfyui(audio_tensor, sample_rate)
            
            
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
                    # Connected audio: save tensor to temporary file for web access
                    try:
                        input_dir = folder_paths.get_input_directory()
                        temp_audio_filename = f"connected_audio_{node_id}.wav"
                        temp_audio_path = os.path.join(input_dir, temp_audio_filename)
                        
                        # Convert tensor to numpy array for soundfile
                        audio_numpy = audio_tensor.cpu().numpy()
                        if audio_numpy.ndim == 1:
                            # Mono audio
                            sf.write(temp_audio_path, audio_numpy, sample_rate)
                        else:
                            # Multi-channel audio - use first channel or average
                            if audio_numpy.shape[0] == 1:
                                sf.write(temp_audio_path, audio_numpy[0], sample_rate)
                            else:
                                # Average multiple channels to mono
                                mono_audio = np.mean(audio_numpy, axis=0)
                                sf.write(temp_audio_path, mono_audio, sample_rate)
                        
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
            
            return (timing_data, visualization_json, analysis_info, processed_audio)
            
        except Exception as e:
            import traceback
            error_msg = f"Audio analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"Full traceback: {traceback.format_exc()}")
            
            # Return error data
            empty_audio = torch.zeros(1, 1000)  # 1 second of silence
            processed_audio = AudioProcessingUtils.format_for_comfyui(empty_audio, 22050)
            
            return (
                f"Error: {error_msg}",
                json.dumps({"error": error_msg, "traceback": traceback.format_exc()}),
                f"Analysis failed: {error_msg}",
                processed_audio
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
        
        # Validate optional inputs
        validated["silence_threshold"] = max(0.001, min(0.1, inputs.get("silence_threshold", 0.01)))
        validated["silence_min_duration"] = max(0.01, min(2.0, inputs.get("silence_min_duration", 0.1)))
        validated["energy_sensitivity"] = max(0.0, min(1.0, inputs.get("energy_sensitivity", 0.5)))
        validated["peak_threshold"] = max(0.001, min(0.5, inputs.get("peak_threshold", 0.02)))
        validated["peak_min_distance"] = max(0.01, min(1.0, inputs.get("peak_min_distance", 0.05)))
        validated["peak_region_size"] = max(0.02, min(1.0, inputs.get("peak_region_size", 0.1)))
        validated["manual_regions"] = inputs.get("manual_regions", "")
        validated["region_labels"] = inputs.get("region_labels", "")
        validated["export_format"] = inputs.get("export_format", "f5tts")
        
        return validated


# Node class mappings for registration
NODE_CLASS_MAPPINGS = {
    "AudioAnalyzerNode": AudioAnalyzerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioAnalyzerNode": "🎵 Audio Analyzer"
}