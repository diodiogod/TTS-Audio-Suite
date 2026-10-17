"""Keep IndexTTS's staged loading and QwenEmotion placement intact."""

from .generic_handler import GenericHandler


class IndexTTSHandler(GenericHandler):
    def partially_load(self, wrapper, device: str) -> bool:
        model = wrapper._model_ref() if wrapper._model_ref else None
        if model is None:
            return False

        # The backend moves its own tensors. Recursing afterward would put all
        # low-VRAM components on the GPU and override QwenEmotion's GPU choice.
        model.to(device)
        return True
