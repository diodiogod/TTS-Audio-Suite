"""
Granite ASR runtime wrapper.
"""

import os
import re
from typing import Dict, Optional

import torch

from engines.granite_asr.prompting import DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE


class GraniteASRRuntime:
    """
    Thin runtime wrapper for Granite speech generation.
    """

    DEFAULT_CHAT_TEMPLATE = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "USER: {{ message['content'] }}\n ASSISTANT:"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
    )

    LANGUAGE_NAMES = {
        "english": "English",
        "french": "French",
        "german": "German",
        "spanish": "Spanish",
        "portuguese": "Portuguese",
        "japanese": "Japanese",
    }

    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self._ensure_chat_template()

    def to(self, device):
        if hasattr(self.model, "to"):
            self.model.to(device)
        self.device = str(device)
        return self

    @property
    def dtype(self):
        return getattr(self.model, "dtype", torch.float32)

    def _ensure_chat_template(self):
        tokenizer = self.processor.tokenizer
        if getattr(tokenizer, "chat_template", None):
            return

        tokenizer_path = getattr(tokenizer, "name_or_path", None)
        if tokenizer_path:
            template_path = os.path.join(tokenizer_path, "chat_template.jinja")
            if os.path.exists(template_path):
                try:
                    with open(template_path, "r", encoding="utf-8") as f:
                        tokenizer.chat_template = f.read()
                    return
                except Exception:
                    pass

        tokenizer.chat_template = self.DEFAULT_CHAT_TEMPLATE

    def _build_prompt(
        self,
        task: str,
        language: Optional[str] = None,
        target_language: Optional[str] = None,
        translate_instruction_override: Optional[str] = None,
    ) -> str:
        if task == "translate":
            target_language = target_language or "English"
            source_language = language or "the spoken source language"
            user_prompt = str(translate_instruction_override or DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE).strip()
            user_prompt = user_prompt.replace("{source_language}", source_language)
            user_prompt = user_prompt.replace("{language}", source_language)
            user_prompt = user_prompt.replace("{target_language}", target_language)

            if "<|audio|>" not in user_prompt:
                user_prompt = f"<|audio|>{user_prompt.lstrip()}"
        else:
            if language:
                user_prompt = f"<|audio|>can you transcribe the {language} speech into a written format?"
            else:
                user_prompt = "<|audio|>can you transcribe the speech into a written format?"

        chat = [{"role": "user", "content": user_prompt}]
        tokenizer = self.processor.tokenizer
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text.strip()

    def _build_generation_kwargs(self, generation_kwargs: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        generation_kwargs = dict(generation_kwargs or {})
        max_new_tokens = int(generation_kwargs.pop("max_new_tokens", 200))
        kwargs: Dict[str, object] = {
            "max_new_tokens": max_new_tokens,
        }

        for key in (
            "do_sample",
            "num_beams",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "length_penalty",
            "no_repeat_ngram_size",
            "early_stopping",
        ):
            if key in generation_kwargs:
                kwargs[key] = generation_kwargs[key]

        return kwargs

    @torch.inference_mode()
    def transcribe(
        self,
        waveform,
        task: str = "transcribe",
        language: Optional[str] = None,
        target_language: Optional[str] = None,
        translate_instruction_override: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, object]] = None,
    ):
        prompt = self._build_prompt(
            task=task,
            language=language,
            target_language=target_language,
            translate_instruction_override=translate_instruction_override,
        )
        generate_kwargs = self._build_generation_kwargs(generation_kwargs)

        model_inputs = self.processor(
            prompt,
            waveform,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(self.device)

        output_ids = self.model.generate(
            **model_inputs,
            **generate_kwargs,
        )

        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = output_ids[:, num_input_tokens:]
        decoded = self.processor.tokenizer.batch_decode(
            new_tokens,
            add_special_tokens=False,
            skip_special_tokens=True,
        )

        text = self._clean_text(decoded[0] if decoded else "")
        return {
            "text": text,
            "language": language,
            "target_language": target_language,
            "task": task,
        }
