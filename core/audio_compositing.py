"""
Audio compositing utilities for F5-TTS editing.
Exact working implementation extracted from f5tts_edit_node.py
Enhanced with configurable crossfade curves and adaptive behavior
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
                              noise_floor_matching: bool = False, dynamic_range_compression: bool = True) -> torch.Tensor:
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
        
        # Ensure same length (pad shorter one with zeros)
        max_length = max(original_audio.shape[-1], generated_audio.shape[-1])
        if original_audio.shape[-1] < max_length:
            padding = max_length - original_audio.shape[-1]
            padding_tensor = torch.zeros(original_audio.shape[0], padding, device=original_audio.device)
            original_audio = torch.cat([original_audio, padding_tensor], dim=-1)
        if generated_audio.shape[-1] < max_length:
            padding = max_length - generated_audio.shape[-1]
            padding_tensor = torch.zeros(generated_audio.shape[0], padding, device=generated_audio.device)
            generated_audio = torch.cat([generated_audio, padding_tensor], dim=-1)
        
        # Start with original audio
        composite_audio = original_audio.clone()
        
        for start_time, end_time in edit_regions:
            # Initialize severe correction flag for this segment
            severe_correction_applied = False
            # Calculate adaptive crossfade if enabled
            if adaptive_crossfade:
                segment_duration = end_time - start_time
                adaptive_crossfade_ms = AudioCompositor._calculate_adaptive_crossfade(segment_duration, crossfade_duration_ms)
                print(f"🔄 ADAPTIVE CROSSFADE: {segment_duration:.3f}s segment -> {adaptive_crossfade_ms}ms crossfade")
            else:
                adaptive_crossfade_ms = crossfade_duration_ms
            
            # Calculate crossfade samples
            crossfade_samples = int(adaptive_crossfade_ms * sample_rate / 1000)
            print(f"🎵 CROSSFADE: {adaptive_crossfade_ms}ms = {crossfade_samples} samples at {sample_rate}Hz for region {start_time:.3f}-{end_time:.3f}s")
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Ensure we don't go beyond audio bounds
            start_sample = max(0, min(start_sample, max_length))
            end_sample = max(start_sample, min(end_sample, max_length))
            
            if start_sample >= end_sample:
                continue
            
            # Extract edited segment from generated audio
            edited_segment = generated_audio[:, start_sample:end_sample]
            
            # Debug audio levels at boundaries for click detection
            if start_sample > 0:
                orig_before = original_audio[:, start_sample-1:start_sample+1].abs().mean().item()
                gen_start = edited_segment[:, 0:2].abs().mean().item()
                print(f"    📊 BOUNDARY LEVELS - Original before: {orig_before:.4f}, Generated start: {gen_start:.4f}, Ratio: {gen_start/orig_before:.2f}x")
            
            if end_sample < original_audio.shape[-1]:
                orig_after = original_audio[:, end_sample-1:end_sample+1].abs().mean().item() 
                gen_end = edited_segment[:, -2:].abs().mean().item()
                print(f"    📊 BOUNDARY LEVELS - Generated end: {gen_end:.4f}, Original after: {orig_after:.4f}, Ratio: {gen_end/orig_after:.2f}x")
            
            # Apply full segment normalization FIRST (before boundary matching)
            # BUT skip if we have severe volume mismatches to avoid double-correction
            if full_segment_normalization:
                # Use the boundary level data we just calculated for aggressive normalization
                if start_sample > 0:
                    # Use the original before level as target
                    orig_before_level = original_audio[:, start_sample-1:start_sample+1].abs().mean().item()
                    gen_start_level = edited_segment[:, 0:2].abs().mean().item()
                    if gen_start_level > 1e-6:
                        target_ratio = orig_before_level / gen_start_level
                        print(f"    📈 VOLUME ANALYSIS: Need {target_ratio:.2f}x correction for {1/target_ratio:.0f}% volume mismatch")
                        
                        # Smart correction based on severity - but more conservative to avoid artifacts
                        if target_ratio > 3.0:  # Severe mismatch (like your 4.32x)
                            # Much more conservative for severe cases to avoid clicks
                            smart_ratio = 1.8  # Reduced from 2.5 to 1.8 
                            edited_segment *= smart_ratio
                            print(f"    🎯 CONSERVATIVE SEVERE CORRECTION: {smart_ratio:.2f}x applied (capped from {target_ratio:.2f}x)")
                            # Mark that we applied severe correction to skip boundary matching
                            severe_correction_applied = True
                        elif 1.5 < target_ratio <= 3.0:  # Moderate mismatch
                            # Apply 60% of correction for moderate cases (reduced from 80%)
                            smart_ratio = 1.0 + (target_ratio - 1.0) * 0.6
                            edited_segment *= smart_ratio
                            print(f"    🎯 MODERATE CORRECTION: {smart_ratio:.2f}x applied ({target_ratio:.2f}x needed)")
                            severe_correction_applied = False
                        elif 0.5 < target_ratio <= 1.5:  # Minor mismatch
                            # Apply gentle correction for minor cases
                            smart_ratio = 1.0 + (target_ratio - 1.0) * 0.5
                            edited_segment *= smart_ratio
                            print(f"    🎯 GENTLE CORRECTION: {smart_ratio:.2f}x applied ({target_ratio:.2f}x needed)")
                            severe_correction_applied = False
                        else:
                            severe_correction_applied = False
                
                elif end_sample < original_audio.shape[-1]:
                    # Use the original after level as target  
                    orig_after_level = original_audio[:, end_sample-1:end_sample+1].abs().mean().item()
                    gen_end_level = edited_segment[:, -2:].abs().mean().item()
                    if gen_end_level > 1e-6:
                        target_ratio = orig_after_level / gen_end_level
                        print(f"    📈 VOLUME ANALYSIS: Need {target_ratio:.2f}x correction for {1/target_ratio:.0f}% volume mismatch")
                        
                        # Smart correction based on severity - but more conservative to avoid artifacts
                        if target_ratio > 3.0:  # Severe mismatch
                            smart_ratio = 1.8  # Reduced from 2.5 to 1.8
                            edited_segment *= smart_ratio
                            print(f"    🎯 CONSERVATIVE SEVERE CORRECTION: {smart_ratio:.2f}x applied (capped from {target_ratio:.2f}x)")
                            severe_correction_applied = True
                        elif 1.5 < target_ratio <= 3.0:  # Moderate mismatch
                            smart_ratio = 1.0 + (target_ratio - 1.0) * 0.6
                            edited_segment *= smart_ratio
                            print(f"    🎯 MODERATE CORRECTION: {smart_ratio:.2f}x applied ({target_ratio:.2f}x needed)")
                            severe_correction_applied = False
                        elif 0.5 < target_ratio <= 1.5:  # Minor mismatch
                            smart_ratio = 1.0 + (target_ratio - 1.0) * 0.5
                            edited_segment *= smart_ratio
                            print(f"    🎯 GENTLE CORRECTION: {smart_ratio:.2f}x applied ({target_ratio:.2f}x needed)")
                            severe_correction_applied = False
                        else:
                            severe_correction_applied = False

            # Apply boundary volume matching if enabled (now after full normalization)
            # Skip if severe correction was already applied to avoid double-correction artifacts
            if boundary_volume_matching and not locals().get('severe_correction_applied', False):
                boundary_samples = min(480, edited_segment.shape[-1] // 4)  # ~20ms at 24kHz
                
                if start_sample > 0 and boundary_samples > 0:
                    orig_before_rms = original_audio[:, max(0, start_sample-boundary_samples):start_sample].square().mean().sqrt()
                    gen_start_rms = edited_segment[:, :boundary_samples].square().mean().sqrt()
                    if gen_start_rms > 1e-6:  # Avoid division by zero
                        volume_ratio = orig_before_rms / gen_start_rms
                        if 0.5 < volume_ratio < 2.0:  # Only adjust reasonable ratios
                            edited_segment[:, :boundary_samples] *= volume_ratio
                            print(f"    🎚️ START VOLUME MATCH: {volume_ratio:.2f}x adjustment")
                
                if end_sample < original_audio.shape[-1] and boundary_samples > 0:
                    orig_after_rms = original_audio[:, end_sample:min(original_audio.shape[-1], end_sample+boundary_samples)].square().mean().sqrt()
                    gen_end_rms = edited_segment[:, -boundary_samples:].square().mean().sqrt()
                    if gen_end_rms > 1e-6:  # Avoid division by zero
                        volume_ratio = orig_after_rms / gen_end_rms
                        if 0.5 < volume_ratio < 2.0:  # Only adjust reasonable ratios
                            edited_segment[:, -boundary_samples:] *= volume_ratio
                            print(f"    🎚️ END VOLUME MATCH: {volume_ratio:.2f}x adjustment")
            
            
            # Apply spectral matching (basic EQ to match frequency characteristics)
            if spectral_matching:
                print(f"    🎼 SPECTRAL MATCHING: Attempting frequency analysis...")
                # This is a simplified implementation - could be enhanced with FFT-based matching
                # Apply high-frequency emphasis/de-emphasis based on original audio characteristics
                try:
                    # Simple high-pass filtering to match speech clarity
                    from scipy import signal
                    nyquist = sample_rate / 2
                    high_freq = 3000  # Hz
                    b, a = signal.butter(2, high_freq / nyquist, btype='high')
                    
                    # Apply to both original context and generated, then match levels
                    if start_sample > 100 and end_sample + 100 < original_audio.shape[-1]:
                        orig_context = original_audio[:, start_sample-100:start_sample+100].cpu().numpy().copy()  # Fix numpy stride issue
                        orig_filtered = signal.filtfilt(b, a, orig_context, axis=-1)
                        orig_high_rms = np.sqrt(np.mean(orig_filtered**2))
                        
                        gen_filtered = signal.filtfilt(b, a, edited_segment.cpu().numpy().copy(), axis=-1)  # Fix numpy stride issue
                        gen_high_rms = np.sqrt(np.mean(gen_filtered**2))
                        
                        if gen_high_rms > 1e-6:
                            spectral_ratio = orig_high_rms / gen_high_rms
                            if 0.5 < spectral_ratio < 2.0:
                                # Apply simple high-frequency adjustment
                                edited_segment_filtered = torch.from_numpy(gen_filtered).to(edited_segment.device)
                                adjustment = (spectral_ratio - 1.0) * 0.1  # Gentle adjustment
                                edited_segment += edited_segment_filtered.float() * adjustment
                                print(f"    🎼 SPECTRAL MATCH: {spectral_ratio:.2f}x high-freq adjustment")
                except Exception as e:
                    print(f"    ⚠️ Spectral matching failed: {e}")
            
            # Apply noise floor matching
            if noise_floor_matching:
                print(f"    🔊 NOISE FLOOR: Analyzing background noise levels...")
                # Calculate noise floor from quiet sections of original audio
                try:
                    # Find low-energy regions in original audio (likely background noise)
                    window_size = sample_rate // 10  # 100ms windows
                    min_noise_level = float('inf')
                    
                    for i in range(0, original_audio.shape[-1] - window_size, window_size):
                        window = original_audio[:, i:i+window_size]
                        window_rms = window.square().mean().sqrt().item()
                        if window_rms < min_noise_level and window_rms > 1e-6:
                            min_noise_level = window_rms
                    
                    # Add subtle noise to generated segment to match noise floor
                    if min_noise_level < float('inf'):
                        gen_noise_level = edited_segment.square().mean().sqrt().item()
                        if gen_noise_level < min_noise_level * 0.5:  # Only if generated is too clean
                            noise_to_add = min_noise_level * 0.3  # Add 30% of detected noise level
                            noise = torch.randn_like(edited_segment) * noise_to_add
                            edited_segment += noise
                            print(f"    🔊 NOISE FLOOR MATCH: Added {noise_to_add:.6f} noise level")
                except Exception as e:
                    print(f"    ⚠️ Noise floor matching failed: {e}")
            
            # Apply dynamic range compression to smooth volume variations
            if dynamic_range_compression:
                print(f"    🎛️ COMPRESSION: Analyzing peaks above threshold...")
                try:
                    # Simple soft compression - reduce peaks gently
                    threshold = 0.7  # Compression threshold
                    ratio = 3.0      # Compression ratio
                    
                    # Calculate compression
                    segment_abs = edited_segment.abs()
                    over_threshold = segment_abs > threshold
                    
                    if over_threshold.any():
                        # Apply soft knee compression
                        compressed_values = threshold + (segment_abs - threshold) / ratio
                        compression_mask = over_threshold.float()
                        edited_segment = edited_segment.sign() * (
                            segment_abs * (1 - compression_mask) + 
                            compressed_values * compression_mask
                        )
                        compression_amount = over_threshold.sum().item() / edited_segment.numel() * 100
                        print(f"    🎛️ COMPRESSION: {compression_amount:.1f}% of samples compressed")
                    else:
                        max_level = segment_abs.max().item()
                        print(f"    🎛️ COMPRESSION: No peaks above {threshold:.1f} threshold (max: {max_level:.3f})")
                except Exception as e:
                    print(f"    ⚠️ Compression failed: {e}")
            
            # Apply crossfade at boundaries to avoid clicks
            if crossfade_samples > 0:
                print(f"  💫 APPLYING CROSSFADE ({crossfade_samples} samples) with {crossfade_curve} curve")
                # Crossfade at start
                if start_sample > 0:
                    fade_start = max(0, start_sample - crossfade_samples)
                    fade_length = start_sample - fade_start
                    if fade_length > 0:
                        print(f"    🔸 START CROSSFADE: samples {fade_start}-{start_sample} (length={fade_length})")
                        # Create fade weights using specified curve
                        fade_out = 1.0 - AudioCompositor._apply_crossfade_curve(fade_length, crossfade_curve, composite_audio.device)
                        fade_in = AudioCompositor._apply_crossfade_curve(fade_length, crossfade_curve, composite_audio.device)
                        
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
                        print(f"    🔹 END CROSSFADE: samples {end_sample}-{fade_end} (length={fade_length})")
                        # Create fade weights using specified curve
                        fade_out = 1.0 - AudioCompositor._apply_crossfade_curve(fade_length, crossfade_curve, composite_audio.device)
                        fade_in = AudioCompositor._apply_crossfade_curve(fade_length, crossfade_curve, composite_audio.device)
                        
                        # Apply crossfade
                        if end_sample < generated_audio.shape[-1]:
                            gen_fade_start = end_sample
                            gen_fade_end = min(fade_end, generated_audio.shape[-1])
                            composite_audio[:, end_sample:gen_fade_end] *= fade_out[:gen_fade_end-end_sample]
                            composite_audio[:, end_sample:gen_fade_end] += generated_audio[:, gen_fade_start:gen_fade_end] * fade_in[:gen_fade_end-end_sample]
            
            else:
                print(f"  ⚡ NO CROSSFADE - Hard cut (0ms crossfade)")
            
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