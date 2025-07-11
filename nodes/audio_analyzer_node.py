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
                    "tooltip": "Path to audio file or drag audio file here",
                    "dynamicPrompts": False
                }),
                "analysis_method": (["silence", "energy", "peaks", "manual"], {
                    "default": "silence",
                    "tooltip": "Method for automatic timing detection"
                }),
                "precision_level": (["seconds", "milliseconds", "samples"], {
                    "default": "milliseconds",
                    "tooltip": "Precision level for timing output"
                }),
                "visualization_points": ("INT", {
                    "default": 2000,
                    "min": 500,
                    "max": 10000,
                    "step": 100,
                    "tooltip": "Number of points for waveform visualization"
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Optional: Audio input from another node"
                }),
                "silence_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Amplitude threshold for silence detection"
                }),
                "silence_min_duration": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Minimum duration for silence regions (seconds)"
                }),
                "energy_sensitivity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Sensitivity for energy-based detection"
                }),
                "manual_regions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Manual timing regions (start,end format, one per line)"
                }),
                "region_labels": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Labels for timing regions (one per line, optional)"
                }),
                "export_format": (["f5tts", "json", "csv"], {
                    "default": "f5tts",
                    "tooltip": "Output format for timing data"
                }),
                "node_id": ("STRING", {
                    "default": "",
                    "tooltip": "Internal node ID for cache management"
                }),
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
    
    def _create_analysis_info(self, audio_tensor: torch.Tensor, sample_rate: int, 
                             regions: List[TimingRegion], method: str) -> str:
        """Create analysis information string."""
        duration = AudioProcessingUtils.get_audio_duration(audio_tensor, sample_rate)
        
        info_lines = [
            f"Audio Analysis Results",
            f"Duration: {duration:.2f} seconds",
            f"Sample Rate: {sample_rate} Hz",
            f"Analysis Method: {method}",
            f"Regions Found: {len(regions)}",
            f"",
            "Timing Regions:"
        ]
        
        for i, region in enumerate(regions):
            region_duration = region.end_time - region.start_time
            info_lines.append(
                f"  {i+1}. {region.label}: {region.start_time:.3f}s - {region.end_time:.3f}s "
                f"(duration: {region_duration:.3f}s, confidence: {region.confidence:.2f})"
            )
        
        return "\n".join(info_lines)
    
    def analyze_audio(self, audio_file, analysis_method="silence", precision_level="milliseconds",
                     visualization_points=2000, audio=None, silence_threshold=0.01, silence_min_duration=0.1,
                     energy_sensitivity=0.5, manual_regions="", region_labels="",
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
            Tuple of (timing_data, visualization_data, analysis_info, processed_audio)
        """
        
        try:
            # Handle audio input - either from file or from input
            if audio is not None:
                # Audio input from another node
                audio_tensor, sample_rate = self._extract_audio_tensor(audio)
            elif audio_file and audio_file.strip():
                # Load audio from file path
                file_path = audio_file.strip()
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
            cache_key = f"{tensor_hash}_{analysis_method}_{silence_threshold}_{silence_min_duration}_{energy_sensitivity}"
            
            # Check cache first
            cached_result = analysis_cache.get(cache_key)
            if cached_result:
                regions = cached_result
                print("📋 Using cached analysis results")
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
                    regions = self.analyzer.extract_timing_regions(audio_tensor, method="peaks")
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
            
            # Format timing data according to export format
            if export_format == "f5tts":
                timing_data = self.analyzer.format_timing_for_f5tts(regions)
            else:
                timing_data = json.dumps(
                    self.analyzer.export_timing_data(regions, export_format),
                    indent=2
                )
            
            # Create analysis info
            analysis_info = self._create_analysis_info(audio_tensor, sample_rate, regions, analysis_method)
            
            # Format visualization data as JSON
            visualization_json = json.dumps(viz_data, indent=2)
            
            # Return processed audio in ComfyUI format
            processed_audio = AudioProcessingUtils.format_for_comfyui(audio_tensor, sample_rate)
            
            
            # Save visualization data to ComfyUI temp directory and copy audio for web access
            try:
                import folder_paths
                import shutil
                
                # Save visualization data
                temp_dir = folder_paths.get_temp_directory()
                temp_file = os.path.join(temp_dir, f"audio_data_{node_id}.json")
                
                with open(temp_file, 'w') as f:
                    json.dump(viz_data, f, indent=2)
                
                print(f"🎵 Audio data saved to temp: {temp_file}")
                
                # Copy audio file to ComfyUI input directory for web access
                if audio_file and audio_file.strip() and os.path.exists(audio_file.strip()):
                    input_dir = folder_paths.get_input_directory()
                    audio_filename = os.path.basename(audio_file.strip())
                    web_audio_path = os.path.join(input_dir, audio_filename)
                    
                    # Copy if not already there or if source is newer
                    if not os.path.exists(web_audio_path) or os.path.getmtime(audio_file.strip()) > os.path.getmtime(web_audio_path):
                        shutil.copy2(audio_file.strip(), web_audio_path)
                        print(f"🎵 Audio file copied for web access: {web_audio_path}")
                
            except Exception as save_error:
                print(f"⚠️ Audio Analyzer data save failed: {save_error}")
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
        validated["analysis_method"] = inputs.get("analysis_method", "silence")
        validated["precision_level"] = inputs.get("precision_level", "milliseconds")
        validated["visualization_points"] = max(500, min(10000, inputs.get("visualization_points", 2000)))
        
        # Validate optional inputs
        validated["silence_threshold"] = max(0.001, min(0.1, inputs.get("silence_threshold", 0.01)))
        validated["silence_min_duration"] = max(0.01, min(2.0, inputs.get("silence_min_duration", 0.1)))
        validated["energy_sensitivity"] = max(0.0, min(1.0, inputs.get("energy_sensitivity", 0.5)))
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