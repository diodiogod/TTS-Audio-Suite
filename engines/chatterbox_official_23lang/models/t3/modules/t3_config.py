from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    def __init__(self, text_tokens_dict_size=704):
        self.start_text_token = 255
        self.stop_text_token = 0
        self.text_tokens_dict_size = text_tokens_dict_size
        self.max_text_tokens = 2048

        self.start_speech_token = 6561
        self.stop_speech_token = 6562
        self.speech_tokens_dict_size = 8194
        self.max_speech_tokens = 4096

        self.llama_config_name = "Llama_520M"
        self.input_pos_emb = "learned"
        self.speech_cond_prompt_len = 150

        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.use_perceiver_resampler = True
        self.emotion_adv = True

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]
    
    @property
    def is_multilingual(self):
        # Both v1 (2352) and v2 (2454) are multilingual, English-only is 704
        return self.text_tokens_dict_size in [2352, 2454]

    @classmethod
    def english_only(cls):
        """Create configuration for English-only TTS model."""
        return cls(text_tokens_dict_size=704)

    @classmethod
    def multilingual(cls, version="v2"):
        """Create configuration for multilingual TTS model.

        Args:
            version: "v1" (2352 tokens) or "v2" (2454 tokens with emotion/sound special tokens)
        """
        if version == "v1":
            return cls(text_tokens_dict_size=2352)  # v1: 2352 tokens (original multilingual)
        else:
            return cls(text_tokens_dict_size=2454)  # v2: 2454 tokens (102 new special tokens)