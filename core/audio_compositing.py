"""
Audio compositing utilities for F5-TTS editing.
Exact working implementation extracted from f5tts_edit_node.py
"""

import torch
from typing import List, Tuple, Optional
import torchaudio


class AudioCompositor:
    """Handles compositing of edited audio with original audio to preserve quality."""
    
    @staticmethod
    def composite_edited_audio(original_audio: torch.Tensor, generated_audio: torch.Tensor, 
                              edit_regions: List[Tuple[float, float]], sample_rate: int, 
                              crossfade_ms: int = 50) -> torch.Tensor:
        """Composite edited audio by preserving original audio outside edit regions"""
        # Ensure both audios are same shape
        if original_audio.dim() > 1:
            original_audio = torch.mean(original_audio, dim=0, keepdim=True)
        if generated_audio.dim() > 1:
            generated_audio = torch.mean(generated_audio, dim=0, keepdim=True)
        
        # Ensure same length (pad shorter one with zeros)
        max_length = max(original_audio.shape[-1], generated_audio.shape[-1])
        if original_audio.shape[-1] < max_length:
            padding = max_length - original_audio.shape[-1]
            original_audio = torch.cat([original_audio, torch.zeros(1, padding, device=original_audio.device)], dim=-1)
        if generated_audio.shape[-1] < max_length:
            padding = max_length - generated_audio.shape[-1]
            generated_audio = torch.cat([generated_audio, torch.zeros(1, padding, device=generated_audio.device)], dim=-1)
        
        # Start with original audio
        composite_audio = original_audio.clone()
        
        # Calculate crossfade samples
        crossfade_samples = int(crossfade_ms * sample_rate / 1000)
        
        for start_time, end_time in edit_regions:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Ensure we don't go beyond audio bounds
            start_sample = max(0, min(start_sample, max_length))
            end_sample = max(start_sample, min(end_sample, max_length))
            
            if start_sample >= end_sample:
                continue
            
            # Extract edited segment from generated audio
            edited_segment = generated_audio[:, start_sample:end_sample]
            
            # Apply crossfade at boundaries to avoid clicks
            if crossfade_samples > 0:
                # Crossfade at start
                if start_sample > 0:
                    fade_start = max(0, start_sample - crossfade_samples)
                    fade_length = start_sample - fade_start
                    if fade_length > 0:
                        # Create fade weights
                        fade_out = torch.linspace(1.0, 0.0, fade_length, device=composite_audio.device)
                        fade_in = torch.linspace(0.0, 1.0, fade_length, device=composite_audio.device)
                        
                        # Apply crossfade
                        composite_audio[:, fade_start:start_sample] *= fade_out
                        if fade_start < generated_audio.shape[-1]:
                            gen_fade_end = min(start_sample, generated_audio.shape[-1])
                            composite_audio[:, fade_start:gen_fade_end] += generated_audio[:, fade_start:gen_fade_end] * fade_in[:gen_fade_end-fade_start]
                
                # Crossfade at end
                if end_sample < max_length:
                    fade_end = min(max_length, end_sample + crossfade_samples)
                    fade_length = fade_end - end_sample
                    if fade_length > 0:
                        # Create fade weights
                        fade_out = torch.linspace(1.0, 0.0, fade_length, device=composite_audio.device)
                        fade_in = torch.linspace(0.0, 1.0, fade_length, device=composite_audio.device)
                        
                        # Apply crossfade
                        if end_sample < generated_audio.shape[-1]:
                            gen_fade_start = end_sample
                            gen_fade_end = min(fade_end, generated_audio.shape[-1])
                            composite_audio[:, end_sample:gen_fade_end] *= fade_out[:gen_fade_end-end_sample]
                            composite_audio[:, end_sample:gen_fade_end] += generated_audio[:, gen_fade_start:gen_fade_end] * fade_in[:gen_fade_end-end_sample]
            
            # Replace the main edit region
            composite_audio[:, start_sample:end_sample] = edited_segment
        
        return composite_audio


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