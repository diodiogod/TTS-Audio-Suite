from dataclasses import dataclass, field
from typing import Dict, Any
from ..llama_configs import LLAMA_CONFIGS


@dataclass
class T3Config:
    # Text tokens
    start_text_token: int = 255
    stop_text_token: int = 0
    text_tokens_dict_size: int = 704
    max_text_tokens: int = 2048

    # Speech tokens
    start_speech_token: int = 6561
    stop_speech_token: int = 6562
    speech_tokens_dict_size: int = 8194
    max_speech_tokens: int = 4096

    # Model settings
    llama_config_name: str = "Llama_520M"
    input_pos_emb: str = "learned"
    speech_cond_prompt_len: int = 150

    # T3CondEnc settings
    encoder_type: str = "voice_encoder"
    speaker_embed_size: int = 256
    use_perceiver_resampler: bool = True
    emotion_adv: bool = True

    # Model configuration with defaults
    model_cfg: Dict[str, Any] = field(default_factory=lambda: {
        "attn_implementation": "eager",
        "output_attentions": False,
        "use_cache": True,
        "return_dict": True
    })

    @property
    def n_channels(self) -> int:
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]
