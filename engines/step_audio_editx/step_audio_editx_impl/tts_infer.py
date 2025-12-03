import os
import argparse
import torch

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from datetime import datetime
import torchaudio
import librosa
import soundfile as sf

# Project imports
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from model_loader import ModelSource
from config.edit_config import get_supported_edit_types


# Save audio to temporary directory
def save_audio(filename, audio_data, sr, output_dir):
    """Save audio data to a temporary file with timestamp"""
    save_path = os.path.join(output_dir, f"{filename}.wav")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        if isinstance(audio_data, torch.Tensor):
            torchaudio.save(save_path, audio_data, sr)
        else:
            sf.write(save_path, audio_data, sr)
        logger.info(f"Audio saved to: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        raise

    return save_path


class StepAudioEditX:
    """Audio editing and voice cloning local inference class"""

    def __init__(self, args):
        self.args = args
        self.edit_type_list = list(get_supported_edit_types().keys())

    def history_messages_to_show(self, messages):
        show_msgs = []
        for message in messages:
            edit_type = message['edit_type']
            edit_info = message['edit_info']
            source_text = message['source_text']
            target_text = message['target_text']
            raw_audio_part = message['raw_wave']
            edit_audio_part = message['edit_wave']
            type_str = f"{edit_type}-{edit_info}" if edit_info is not None else f"{edit_type}"
            show_msgs.extend([
                {"role": "user", "content": f"任务类型：{type_str}\n文本：{source_text}"},
                {"role": "user", "content": raw_audio_part},
                {"role": "assistant", "content": f"输出音频：\n文本：{target_text}"},
                {"role": "assistant", "content": edit_audio_part}
            ])
        return show_msgs

    def generate_clone(self, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, state, filename_out):
        """Generate cloned audio"""
        logger.info("Starting voice cloning process")
        state['history_audio'] = []
        state['history_messages'] = []

        # Input validation
        if not prompt_text_input or prompt_text_input.strip() == "":
            error_msg = "[Error] Uploaded text cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not generated_text or generated_text.strip() == "":
            error_msg = "[Error] Clone content cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if edit_type != "clone":
            error_msg = "[Error] CLONE button must use clone task."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

        try:
            # Use common_tts_engine for cloning
            output_audio, output_sr = common_tts_engine.clone(
                prompt_audio_input, prompt_text_input, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": prompt_text_input,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                audio_save_path = save_audio(filename_out, audio_numpy, output_sr, self.args.output_dir)
                state["history_audio"].append((output_sr, audio_save_path, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                logger.info("Voice cloning completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Clone failed"
                logger.error(error_msg)
                return [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Clone failed: {str(e)}"
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        
    def generate_edit(self, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, state, filename_out):
        """Generate edited audio"""
        logger.info("Starting audio editing process")

        # Input validation
        if edit_type not in ['denoise', 'vad'] and (not prompt_text_input or prompt_text_input.strip() == ""):
            error_msg = "[Error] Uploaded text cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

        try:
            # Determine which audio to use
            if len(state["history_audio"]) == 0:
                # First edit - use uploaded audio
                audio_to_edit = prompt_audio_input
                text_to_use = prompt_text_input
                logger.debug("Using prompt audio, no history found")
            else:
                # Use previous edited audio - save it to temp file first
                _, audio_save_path, previous_text = state["history_audio"][-1]
                audio_to_edit = audio_save_path
                text_to_use = previous_text
                logger.debug(f"Using previous audio from history, count: {len(state['history_audio'])}")

            # For para-linguistic, use generated_text; otherwise use source text
            if edit_type not in {"paralinguistic"}:
                generated_text = text_to_use

            # Use common_tts_engine for editing
            output_audio, output_sr = common_tts_engine.edit(
                audio_to_edit, text_to_use, edit_type, edit_info, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                if len(state["history_audio"]) == 0:
                    input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)
                else:
                    input_sample_rate, input_audio_data_numpy, _ = state["history_audio"][-1]

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": text_to_use,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                audio_save_path = save_audio(filename_out, audio_numpy, output_sr, self.args.output_dir)
                state["history_audio"].append((output_sr, audio_save_path, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                logger.info("Audio editing completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Edit failed"
                logger.error(error_msg)
                return [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Edit failed: {str(e)}"
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

    def clear_history(self, state):
        """Clear conversation history"""
        state["history_messages"] = []
        state["history_audio"] = []
        return [], state

    def init_state(self):
        """Initialize conversation state"""
        return {
            "history_messages": [],
            "history_audio": []
        }


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Step-Audio-EditX local inference demo")
    parser.add_argument("--model-path", type=str, required=True, help="Model path.")
    parser.add_argument("--output-dir", type=str, default="./output_dir", help="Save path.")

    # Multi-source loading support parameters
    parser.add_argument(
        "--model-source",
        type=str,
        default="auto",
        choices=["auto", "local", "modelscope", "huggingface"],
        help="Model source: auto (detect automatically), local, modelscope, or huggingface"
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
        help="Tokenizer model ID for online loading"
    )
    parser.add_argument(
        "--tts-model-id",
        type=str,
        default=None,
        help="TTS model ID for online loading (if different from model-path)"
    )

    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["int4", "int8"],
        help="Enable quantization for the TTS model to reduce memory usage."
             "Choices: int4 (online), int8 (online)."
             "When quantization is enabled, data types are handled automatically by the quantization library."
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="PyTorch data type for model operations. This setting only applies when quantization is disabled. "
             "When quantization is enabled, data types are managed automatically."
    )

    parser.add_argument(
        "--device-map",
        type=str,
        default="cuda",
        help="Device mapping for model loading (default: cuda)"
    )

    # clone or edit parameters
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="",
        help="prompt text for editing or cloning"
    )

    parser.add_argument(
        "--prompt-audio-path",
        type=str,
        default="",
        help="prompt audio for editing or cloning"
    )

    parser.add_argument(
        "--edit-type",
        type=str,
        choices=["clone", "emotion", "style", "vad", "denoise", "paralinguistic", "speed"],
        default="clone",
        help="Edit type"
    )

    parser.add_argument(
        "--edit-info",
        type=str,
        choices=[
            # default
            '',
            # emotion
            'happy', 'angry', 'sad', 'humour', 'confusion', 'disgusted',
            'empathy', 'embarrass', 'fear', 'surprised', 'excited',
            'depressed', 'coldness', 'admiration', 'remove',
            # style
            'serious', 'arrogant', 'child', 'older', 'girl', 'pure',
            'sister', 'sweet', 'ethereal', 'whisper', 'gentle', 'recite',
            'generous', 'act_coy', 'warm', 'shy', 'comfort', 'authority',
            'chat', 'radio', 'soulful', 'story', 'vivid', 'program',
            'news', 'advertising', 'roar', 'murmur', 'shout', 'deeply', 'loudly',
            'remove', 'exaggerated',
            # speed
            'faster', 'slower', 'more faster', 'more slower'
        ],
        default="",
        help="Edit info/sub-type"
    )

    parser.add_argument(
        "--n-edit-iter",
        type=int,
        default=1,
        help="the number of edit iterations"
    )

    parser.add_argument(    
        "--generated-text",
        type=str,
        default="",
        help="Generated text for cloning or editing(paralinguistic)"
    )

    args = parser.parse_args()

    source_mapping = {
        "auto": ModelSource.AUTO,
        "local": ModelSource.LOCAL,
        "modelscope": ModelSource.MODELSCOPE,
        "huggingface": ModelSource.HUGGINGFACE
    }
    model_source = source_mapping[args.model_source]

    # Map torch dtype string to actual torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_mapping[args.torch_dtype]

    logger.info(f"Loading models with source: {args.model_source}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Tokenizer model ID: {args.tokenizer_model_id}")
    if args.tts_model_id:
        logger.info(f"TTS model ID: {args.tts_model_id}")
    if args.quantization:
        logger.info(f"🔧 {args.quantization.upper()} quantization enabled")

    # Initialize models
    try:
        # Load StepAudioTokenizer
        encoder = StepAudioTokenizer(
            os.path.join(args.model_path, "Step-Audio-Tokenizer"),
            model_source=model_source,
            funasr_model_id=args.tokenizer_model_id
        )
        logger.info("✓ StepAudioTokenizer loaded successfully")

        # Initialize common TTS engine directly
        common_tts_engine = StepAudioTTS(
            os.path.join(args.model_path, "Step-Audio-EditX"),
            encoder,
            model_source=model_source,
            tts_model_id=args.tts_model_id,
            quantization_config=args.quantization,
            torch_dtype=torch_dtype,
            device_map=args.device_map
        )
        logger.info("✓ StepCommonAudioTTS loaded successfully")

    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        logger.error("Please check your model paths and source configuration.")
        exit(1)
    
    # output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create StepAudioEditX instance
    step_audio_editx = StepAudioEditX(args)
    if args.edit_type == "clone":
        filename_out = os.path.basename(args.prompt_audio_path).split('.')[0] + "_cloned"
        _, state = step_audio_editx.generate_clone(
            args.prompt_text,
            args.prompt_audio_path,
            args.generated_text,
            args.edit_type,
            args.edit_info,
            step_audio_editx.init_state(),
            filename_out,
        )

    else:
        state = step_audio_editx.init_state()
        for iter_idx in range(args.n_edit_iter):
            logger.info(f"Starting edit iteration {iter_idx + 1}/{args.n_edit_iter}")
            filename_out = os.path.basename(args.prompt_audio_path).split('.')[0] + f"_edited_iter{iter_idx + 1}"   
            msgs, state = step_audio_editx.generate_edit(
                args.prompt_text,
                args.prompt_audio_path,
                args.generated_text,
                args.edit_type,
                args.edit_info,
                state,
                filename_out,
            )