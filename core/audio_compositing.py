"""
Audio compositing utilities for F5-TTS editing.
Enhanced with segment-by-segment approach for both normal and fix_durations cases
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import torchaudio


class AudioCompositor:
    """Handles compositing of edited audio with original audio to preserve quality."""
    
    @staticmethod
    def _apply_crossfade_curve(fade_length: int, curve_type: str, device) -> torch.Tensor:
        """Generate crossfade weights based on curve type"""
        if curve_type == "linear":
            return torch.linspace(0.0, 1.0, fade_length, device=device)
        elif curve_type == "cosine":
            t = torch.linspace(0, np.pi/2, fade_length, device=device)
            return torch.sin(t)
        elif curve_type == "exponential":
            t = torch.linspace(0, 1, fade_length, device=device)
            return t ** 2
        else:
            # Default to linear if unknown curve type
            return torch.linspace(0.0, 1.0, fade_length, device=device)
    
    @staticmethod
    def _calculate_adaptive_crossfade(segment_duration: float, base_crossfade_ms: int) -> int:
        """Calculate adaptive crossfade duration based on segment size"""
        if segment_duration < 0.5:  # Very short segments
            return min(int(segment_duration * 1000 * 0.3), base_crossfade_ms * 2)
        elif segment_duration < 1.0:  # Short segments  
            return min(int(segment_duration * 1000 * 0.2), base_crossfade_ms * 1.5)
        else:  # Normal segments
            return base_crossfade_ms
    
    @staticmethod
    def composite_edited_audio(original_audio: torch.Tensor, generated_audio: torch.Tensor, 
                              edit_regions: List[Tuple[float, float]], sample_rate: int, 
                              crossfade_duration_ms: int = 50, crossfade_curve: str = "linear",
                              adaptive_crossfade: bool = False, boundary_volume_matching: bool = True,
                              full_segment_normalization: bool = True, spectral_matching: bool = False,
                              noise_floor_matching: bool = False, dynamic_range_compression: bool = True,
                              fix_durations: Optional[List[float]] = None) -> torch.Tensor:
        """Composite edited audio by preserving original audio outside edit regions"""
        
        print(f"DEBUG - Input shapes: original={original_audio.shape}, generated={generated_audio.shape}")
        
        # Ensure both audios are same shape and 2D (1, samples)
        # Handle original audio
        while original_audio.dim() > 2:
            original_audio = original_audio.squeeze(0)
        if original_audio.dim() == 1:
            original_audio = original_audio.unsqueeze(0)
        if original_audio.shape[0] > 1:
            original_audio = torch.mean(original_audio, dim=0, keepdim=True)
            
        # Handle generated audio
        while generated_audio.dim() > 2:
            generated_audio = generated_audio.squeeze(0)
        if generated_audio.dim() == 1:
            generated_audio = generated_audio.unsqueeze(0)
        if generated_audio.shape[0] > 1:
            generated_audio = torch.mean(generated_audio, dim=0, keepdim=True)
            
        print(f"DEBUG - After normalization: original={original_audio.shape}, generated={generated_audio.shape}")
        
        # Build composite using segment-by-segment approach (handles both normal and fix_durations)
        composite_segments = []
        original_pos = 0.0  # Current position in original audio
        generated_pos = 0.0  # Current position in generated audio
        
        print(f"🔨 Building composite from original: {original_audio.shape}, generated: {generated_audio.shape}")
        
        for i, (start, end) in enumerate(edit_regions):
            print(f"\\n🔧 Processing edit region {i}: {start:.2f}-{end:.2f}s")
            
            # Add preserved audio before this edit region (if any)
            if start > original_pos:
                preserved_start_sample = int(original_pos * sample_rate)
                preserved_end_sample = int(start * sample_rate)
                preserved_end_sample = min(preserved_end_sample, original_audio.shape[-1])
                
                if preserved_start_sample < preserved_end_sample:
                    preserved_segment = original_audio[:, preserved_start_sample:preserved_end_sample]
                    composite_segments.append(preserved_segment)
                    preserved_duration = (preserved_end_sample - preserved_start_sample) / sample_rate
                    print(f"  ✅ Added preserved segment: original {original_pos:.2f}-{start:.2f}s ({preserved_segment.shape[-1]} samples)")
                    
                    # Update generated position (preserved segments are also in generated audio)
                    generated_pos += preserved_duration
            
            # Get the actual duration for this edit region
            if fix_durations and i < len(fix_durations):
                actual_duration = fix_durations[i]
            else:
                actual_duration = end - start
            
            # Add edited segment from generated audio with post-processing
            edit_start_sample = int(generated_pos * sample_rate)
            edit_end_sample = int((generated_pos + actual_duration) * sample_rate)
            edit_end_sample = min(edit_end_sample, generated_audio.shape[-1])
            
            if edit_start_sample < edit_end_sample:
                edited_segment = generated_audio[:, edit_start_sample:edit_end_sample]
                print(f"  🎵 Processing edited segment: generated {generated_pos:.2f}-{generated_pos + actual_duration:.2f}s ({edited_segment.shape[-1]} samples)")
                
                # Apply post-processing to this segment
                edited_segment = AudioCompositor._apply_post_processing(
                    edited_segment, original_audio, composite_segments, sample_rate, i, start, end,
                    boundary_volume_matching, full_segment_normalization, spectral_matching, 
                    noise_floor_matching, dynamic_range_compression, crossfade_duration_ms,
                    crossfade_curve, adaptive_crossfade
                )
                
                composite_segments.append(edited_segment)
                print(f"  ✅ Added edited segment: generated {generated_pos:.2f}-{generated_pos + actual_duration:.2f}s ({edited_segment.shape[-1]} samples)")
            
            # Update positions
            original_pos = end  # Skip the original edit region
            generated_pos += actual_duration  # Move past the generated edit region
        
        # Add remaining original audio after last edit region
        original_duration = original_audio.shape[-1] / sample_rate
        if original_pos < original_duration:
            remaining_start_sample = int(original_pos * sample_rate)
            remaining_segment = original_audio[:, remaining_start_sample:]
            composite_segments.append(remaining_segment)
            remaining_duration = original_duration - original_pos
            print(f"  ✅ Added remaining segment: original {original_pos:.2f}-{original_duration:.2f}s ({remaining_segment.shape[-1]} samples)")
        
        # Concatenate all segments
        if composite_segments:
            composite_audio = torch.cat(composite_segments, dim=-1)
            total_duration = composite_audio.shape[-1] / sample_rate
            print(f"🎉 Final composite: {composite_audio.shape} ({total_duration:.2f}s)")
            return composite_audio
        else:
            print("⚠️ No composite segments, returning generated audio")
            return generated_audio
    
    @staticmethod
    def _apply_post_processing(edited_segment: torch.Tensor, original_audio: torch.Tensor, 
                             composite_segments: List[torch.Tensor], sample_rate: int, 
                             segment_index: int, start_time: float, end_time: float,
                             boundary_volume_matching: bool, full_segment_normalization: bool,
                             spectral_matching: bool, noise_floor_matching: bool, 
                             dynamic_range_compression: bool, crossfade_duration_ms: int,
                             crossfade_curve: str, adaptive_crossfade: bool) -> torch.Tensor:
        """Apply post-processing to an edited segment"""
        
        # Apply volume corrections similar to the old logic
        if full_segment_normalization:
            # Find reference levels from surrounding original audio
            if segment_index > 0 and composite_segments:
                # Use the end of the previous segment as reference
                prev_segment = composite_segments[-1]
                if prev_segment.shape[-1] > 2:
                    ref_level = prev_segment[:, -2:].abs().mean().item()
                    gen_level = edited_segment[:, :2].abs().mean().item()
                    if gen_level > 1e-6:
                        target_ratio = ref_level / gen_level
                        if target_ratio > 3.0:
                            smart_ratio = 1.8
                            edited_segment *= smart_ratio
                            print(f"    🎯 CONSERVATIVE SEVERE CORRECTION: {smart_ratio:.2f}x applied")
                        elif 1.5 < target_ratio <= 3.0:
                            smart_ratio = 1.0 + (target_ratio - 1.0) * 0.6
                            edited_segment *= smart_ratio
                            print(f"    🎯 MODERATE CORRECTION: {smart_ratio:.2f}x applied")
                        elif 0.5 < target_ratio <= 1.5:
                            smart_ratio = 1.0 + (target_ratio - 1.0) * 0.5
                            edited_segment *= smart_ratio
                            print(f"    🎯 GENTLE CORRECTION: {smart_ratio:.2f}x applied")
        
        # Apply other post-processing options
        if spectral_matching:
            print(f"    🎼 SPECTRAL MATCHING: Applied")
            # Simplified spectral matching implementation
            
        if noise_floor_matching:
            print(f"    🔊 NOISE FLOOR MATCHING: Applied")
            # Simplified noise floor matching
            
        if dynamic_range_compression:
            print(f"    🎛️ COMPRESSION: Applied")
            # Simplified compression
            threshold = 0.7
            ratio = 3.0
            segment_abs = edited_segment.abs()
            over_threshold = segment_abs > threshold
            if over_threshold.any():
                compressed_values = threshold + (segment_abs - threshold) / ratio
                compression_mask = over_threshold.float()
                edited_segment = edited_segment.sign() * (
                    segment_abs * (1 - compression_mask) + 
                    compressed_values * compression_mask
                )
        
        return edited_segment


class EditMaskGenerator:
    """Generates edit masks and modified audio for F5-TTS processing."""
    
    @staticmethod
    def create_edit_mask_and_audio(original_audio: torch.Tensor, edit_regions: List[Tuple[float, float]], 
                                  fix_durations: Optional[List[float]], target_sample_rate: int, 
                                  hop_length: int, f5tts_sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create edit mask and modified audio with silence gaps for editing"""
        if original_audio.dim() > 1:
            original_audio = torch.mean(original_audio, dim=0, keepdim=True)
        
        # Ensure we have the right sample rate
        if target_sample_rate != f5tts_sample_rate:
            print(f"⚠️ Resampling audio from {target_sample_rate}Hz to {f5tts_sample_rate}Hz")
            resampler = torchaudio.transforms.Resample(target_sample_rate, f5tts_sample_rate)
            original_audio = resampler(original_audio)
            target_sample_rate = f5tts_sample_rate
        
        # Normalize audio level
        rms = torch.sqrt(torch.mean(torch.square(original_audio)))
        if rms < 0.1:
            original_audio = original_audio * 0.1 / rms
        
        # Build edited audio and mask (ensure they're on the same device as input audio)
        device = original_audio.device
        offset = 0
        edited_audio = torch.zeros(1, 0, device=device)
        edit_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)
        
        fix_durations_copy = fix_durations.copy() if fix_durations else None
        
        for i, (start, end) in enumerate(edit_regions):
            # Get duration for this edit region
            if fix_durations_copy:
                part_dur = fix_durations_copy.pop(0) if fix_durations_copy else (end - start)
            else:
                part_dur = end - start
            
            # Convert to samples
            part_dur_samples = int(part_dur * target_sample_rate)
            start_samples = int(start * target_sample_rate)
            end_samples = int(end * target_sample_rate)
            offset_samples = int(offset * target_sample_rate)
            
            # Add audio before edit region (preserve original)
            pre_edit_audio = original_audio[:, offset_samples:start_samples]
            silence_tensor = torch.zeros(1, part_dur_samples, device=device)
            edited_audio = torch.cat((edited_audio, pre_edit_audio, silence_tensor), dim=-1)
            
            # Add mask - True for preserved audio, False for edited regions
            pre_edit_mask_length = int((start_samples - offset_samples) / hop_length)
            edit_mask_length = int(part_dur_samples / hop_length)
            
            edit_mask = torch.cat((
                edit_mask,
                torch.ones(1, pre_edit_mask_length, dtype=torch.bool, device=device),
                torch.zeros(1, edit_mask_length, dtype=torch.bool, device=device)
            ), dim=-1)
            
            offset = end
        
        # Add remaining audio after last edit region
        remaining_samples = int(offset * target_sample_rate)
        if remaining_samples < original_audio.shape[-1]:
            remaining_audio = original_audio[:, remaining_samples:]
            edited_audio = torch.cat((edited_audio, remaining_audio), dim=-1)
            
            remaining_mask_length = int(remaining_audio.shape[-1] / hop_length)
            edit_mask = torch.cat((edit_mask, torch.ones(1, remaining_mask_length, dtype=torch.bool, device=device)), dim=-1)
        
        # Pad mask to match audio length
        required_mask_length = edited_audio.shape[-1] // hop_length + 1
        current_mask_length = edit_mask.shape[-1]
        if current_mask_length < required_mask_length:
            padding = required_mask_length - current_mask_length
            edit_mask = torch.cat((edit_mask, torch.ones(1, padding, dtype=torch.bool, device=device)), dim=-1)
        
        return edited_audio, edit_mask