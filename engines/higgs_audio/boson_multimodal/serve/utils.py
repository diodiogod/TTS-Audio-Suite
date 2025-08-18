import uuid
import base64
import re
import regex
from typing import AsyncGenerator, Union
import io
from pydub import AudioSegment
import torch
import numpy as np
from functools import lru_cache

from ..audio_processing.higgs_audio_tokenizer import HiggsAudioTokenizer


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


async def async_generator_wrap(first_element, gen: AsyncGenerator):
    """Wrap an async generator with the first element."""
    yield first_element
    async for item in gen:
        yield item


@lru_cache(maxsize=50)
def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format."""
    # Read the MP3 file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


def pcm16_to_target_format(
    np_audio: np.ndarray,
    sample_rate: int,
    bit_depth: int,
    channels: int,
    format: str,
    target_rate: int,
):
    wav_audio = AudioSegment(
        np_audio.tobytes(),
        frame_rate=sample_rate,
        sample_width=bit_depth // 8,
        channels=channels,
    )
    if target_rate is not None and target_rate != sample_rate:
        wav_audio = wav_audio.set_frame_rate(target_rate)

    # Convert WAV to MP3
    target_io = io.BytesIO()
    wav_audio.export(target_io, format=format)
    target_io.seek(0)

    return target_io


chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]+")


def contains_chinese(text: str):
    return bool(chinese_char_pattern.search(text))


# remove blank between chinese character
def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if (text[i + 1].isascii() and text[i + 1] != " ") and (text[i - 1].isascii() and text[i - 1] != " "):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def replace_corner_mark(text: str):
    text = text.replace("²", "平方")
    text = text.replace("³", "立方")
    return text


# remove meaningless symbol
def remove_bracket(text: str):
    text = text.replace("（", "").replace("）", "")
    text = text.replace("【", "").replace("】", "")
    text = text.replace("`", "").replace("`", "")
    text = text.replace("——", " ")
    return text


# split paragrah logic：
# 1. per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
# 2. cal sentence len according to lang
# 3. split sentence according to puncatation
def split_paragraph(text: str, tokenize, lang="zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False):
    def calc_utt_length(_text: str):
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    if lang == "zh":
        pounc = ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"]
    else:
        pounc = [".", "?", "!", ";", ":"]
    if comma_split:
        pounc.extend(["，", ","])

    if text[-1] not in pounc:
        if lang == "zh":
            text += "。"
        else:
            text += "."

    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st:i]) > 0:
                utts.append(text[st:i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', "”"]:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1

    final_utts = []
    cur_utt = ""
    for utt in utts:
        if calc_utt_length(cur_utt + utt) > token_max_n and calc_utt_length(cur_utt) > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts


def is_only_punctuation(text: str):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r"^[\p{P}\p{S}]*$"
    return bool(regex.fullmatch(punctuation_pattern, text))


# spell Arabic numerals
def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st:i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return "".join(new_text)


def remove_emoji(text: str):
    # Pattern to match emojis and their modifiers
    # - Standard emoji range
    # - Zero-width joiners (U+200D)
    # - Variation selectors (U+FE0F, U+FE0E)
    # - Skin tone modifiers (U+1F3FB to U+1F3FF)
    emoji_pattern = re.compile(
        r"["
        r"\U00010000-\U0010FFFF"  # Standard emoji range
        r"\u200D"  # Zero-width joiner
        r"\uFE0F\uFE0E"  # Variation selectors
        r"\U0001F3FB-\U0001F3FF"  # Skin tone modifiers
        r"]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def remove_repeated_punctuations(text, punctuations):
    if len(punctuations) == 0:
        return text
    pattern = f"[{re.escape(''.join(punctuations))}]"  # Create regex pattern for given punctuations
    return re.sub(rf"({pattern})\1+", r"\1", text)


def full_to_half_width(text: str) -> str:
    """Convert full-width punctuation to half-width in a given string."""
    full_width = "！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～"
    half_width = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    trans_table = str.maketrans(full_width, half_width)
    return text.translate(trans_table)


def split_interleaved_delayed_audios(
    audio_data: Union[list[list[int]], torch.Tensor],
    audio_tokenizer: HiggsAudioTokenizer,
    audio_stream_eos_id: int,
) -> list[tuple[list[list[int]], torch.Tensor]]:
    separator = [audio_stream_eos_id] * audio_tokenizer.num_codebooks

    # Convert separator to numpy array if audio_data is numpy array
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.transpose(1, 0)
        separator = torch.tensor(separator)
        # Find the indices where the rows equal the separator
        split_indices = torch.where(torch.all(audio_data == separator, dim=1))[0]
        start = 0
        groups = []
        for idx in split_indices:
            groups.append(audio_data[start:idx].transpose(1, 0))
            start = idx + 1
        if start < len(audio_data):
            groups.append(audio_data[start:].transpose(1, 0))
    else:
        groups = []
        current = []
        for row in audio_data:
            current.append(row)

            if row == separator:
                groups.append(current)
                current = []

        # Don't forget the last group if there's no trailing separator
        if current:
            groups.append(current)

    return groups
