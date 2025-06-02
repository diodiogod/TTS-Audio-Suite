import torch
import torchaudio
import os
import tempfile
from pathlib import Path

try:
    import folder_paths
    OUTPUT_DIR = os.path.join(folder_paths.output_directory, "audio")
except ImportError:
    OUTPUT_DIR = str(Path.home() / "ComfyUI" / "output" / "audio")

os.makedirs(OUTPUT_DIR, exist_ok=True)

class ChatterboxGenerate:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CHATTERBOX_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test of the Chatterbox TTS system."}),
                "exaggeration": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1}),
                "cfg_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.55, "min": 0.1, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "filename": ("STRING", {"default": "chatterbox_output"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "ChatterBox"
    OUTPUT_NODE = True

    def generate_speech(self, model, text, exaggeration=0.5, cfg_weight=0.5, temperature=0.8, reference_audio=None, filename="chatterbox_output"):
        audio_tensor = None  # Initialize to avoid UnboundLocalError
        try:
            # Prepare audio prompt if provided
            audio_prompt_path = None
            if reference_audio is not None:
                print(f"Reference audio type: {type(reference_audio)}")
                print(f"Reference audio content: {reference_audio if isinstance(reference_audio, dict) else 'tensor'}")
                
                # Handle ComfyUI audio format (usually a dict with 'waveform' and 'sample_rate')
                if isinstance(reference_audio, dict):
                    if 'waveform' in reference_audio:
                        audio_data = reference_audio['waveform']
                        sample_rate = reference_audio.get('sample_rate', model.sr)
                    else:
                        print(f"Warning: Unknown audio dict format: {reference_audio.keys()}")
                        audio_data = None
                elif torch.is_tensor(reference_audio):
                    audio_data = reference_audio
                    sample_rate = model.sr
                else:
                    print(f"Warning: Unknown audio format: {type(reference_audio)}")
                    audio_data = None
                
                if audio_data is not None:
                    # Save audio prompt to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        audio_prompt_path = tmp_file.name
                        
                        # Ensure audio_data is 2D for torchaudio.save
                        if audio_data.dim() == 1:
                            audio_data = audio_data.unsqueeze(0)
                        elif audio_data.dim() > 2:
                            audio_data = audio_data.squeeze()
                            if audio_data.dim() == 1:
                                audio_data = audio_data.unsqueeze(0)
                        
                        torchaudio.save(audio_prompt_path, audio_data.float(), sample_rate)
                        print(f"Saved reference audio to: {audio_prompt_path}")
            
            # Generate speech
            print(f"Generating speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            audio_tensor = model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
            
            # Fix tensor shape: squeeze to remove extra dimensions and ensure 2D (channels, samples)
            print(f"Original audio tensor shape: {audio_tensor.shape}")
            
            # Remove any extra dimensions and ensure we have (channels, samples)
            audio_tensor = audio_tensor.squeeze()  # Remove singleton dimensions
            
            # If 1D, add channel dimension to make it (1, samples)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            # If still too many dimensions, take the last 2
            elif audio_tensor.dim() > 2:
                audio_tensor = audio_tensor.view(-1, audio_tensor.shape[-1])
                if audio_tensor.shape[0] > 1:
                    audio_tensor = audio_tensor[0:1]  # Take only first channel
            
            print(f"Fixed audio tensor shape: {audio_tensor.shape}")
            
            # Save to output directory
            if not filename.endswith('.wav'):
                filename += '.wav'
            
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            # Ensure tensor is in correct format for saving (2D: channels, samples)
            save_tensor = audio_tensor.squeeze(0) if audio_tensor.dim() == 3 else audio_tensor
            save_tensor = save_tensor.float()  # Ensure float type
            
            torchaudio.save(
                output_path,
                save_tensor,
                model.sr
            )
            
            print(f"Audio saved to: {output_path}")
            
            # Clean up temporary audio prompt file
            if audio_prompt_path and os.path.exists(audio_prompt_path):
                os.unlink(audio_prompt_path)
            
            # Return audio in ComfyUI format: {"waveform": tensor, "sample_rate": int}
            # Ensure tensor has batch dimension: [batch, channels, samples]
            if audio_tensor.dim() == 2:  # [channels, samples]
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension -> [1, channels, samples]
            
            audio_output = {
                "waveform": audio_tensor,
                "sample_rate": model.sr
            }
            
            print(f"Audio output format: waveform shape={audio_output['waveform'].shape}, sample_rate={audio_output['sample_rate']}")
            
            return (audio_output,)
            
        except Exception as e:
            print(f"Error in speech generation: {e}")
            if audio_tensor is not None:
                print(f"Audio tensor info: shape={getattr(audio_tensor, 'shape', 'N/A')}, dtype={getattr(audio_tensor, 'dtype', 'N/A')}")
            else:
                print("Audio tensor was not created")
            
            # Clean up temporary audio prompt file if it exists
            if 'audio_prompt_path' in locals() and audio_prompt_path and os.path.exists(audio_prompt_path):
                os.unlink(audio_prompt_path)
            
            raise e