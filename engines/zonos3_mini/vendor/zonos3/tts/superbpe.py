"""Frozen SuperBPE inference convention, identical to the training loader.

Phoneme-token checkpoints (lingua data.phoneme_tokens_path, text_extra_tokens rows after the text
padding row) add ``phonemes`` to the spec: a pronunciation group in the text, written
``open_marker`` + IPA + ``close_marker`` (U+E000 / U+E001), encodes as open_id, one id per IPA
code point of ``symbols`` (open_id + 2 + index) and close_id, the text between groups as
SuperBPE on its own -- lingua apps.zonos.text_tokenization._phoneme_payload_ids.

Start-token checkpoints (lingua data.start_token_probability) add ``start`` (id, marker U+E002): a text that begins
with the marker starts with that id, which asks for speech from the first generated frame (lingua
apps.zonos.start_token). Pause-token checkpoints (lingua data.pause_token_probability) add ``pauses`` (ids, markers
U+E003 + j, nominal seconds): each marker after a word is one id asking for a pause of roughly that length, the text
between markers encoded on its own (lingua apps.zonos.text_tokenization._pause_payload_ids).

The phoneme rows, then the start row, then the pause rows are the last extra rows (each block optional, the three
contiguous): right after the padding row, or after the speech-edit marker rows of an edit checkpoint (the first of
them at text_vocab + 1 + markers).
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from zonos3.utils.hashing import sha256_bytes

PHONEME_OPEN, PHONEME_CLOSE = "\ue000", "\ue001"  # lingua apps.zonos.phonemes markers (the spec carries its own)
START_MARKER = "\ue002"  # lingua apps.zonos.start_token.START_MARKER
PAUSE_MARKER_RE = re.compile("[\ue003-\ue00a]")  # lingua apps.zonos.pause_tokens.PAUSE_MARKER_RE (U+E003 + bucket)
NO_START_TOKEN = "This checkpoint has no start token: remove the U+E002 start marker"
NO_PAUSE_TOKENS = "This checkpoint has no pause tokens: remove the U+E003-U+E00A pause markers"
START_NOT_FIRST = "The start marker (U+E002) must be the first character of the text, once"


@lru_cache(maxsize=8)
def tokenizer(path, sha, size):
    from tokenizers import Tokenizer

    raw = Path(path).read_bytes()
    if sha256_bytes(raw) != sha:
        raise ValueError("SuperBPE SHA mismatch")
    t = Tokenizer.from_str(raw.decode())
    if t.get_vocab_size(with_added_tokens=True) != size or t.padding is not None or t.truncation is not None:
        raise ValueError("SuperBPE artifact mismatch")
    return t


def validate_spec(spec, text_vocab):
    if (spec["kind"], spec["offset"], spec["bos"], spec["eos"]) != ("superbpe", 4, 2, 3) or text_vocab != spec[
        "vocab_size"
    ] + 4:
        raise ValueError("SuperBPE ID convention mismatch")
    tokenizer(spec["path"], spec["sha256"], spec["vocab_size"])
    phonemes = spec.get("phonemes")
    if phonemes is not None:
        symbols = phonemes["symbols"]
        if phonemes["open_id"] < text_vocab + 1 or phonemes["close_id"] != phonemes["open_id"] + 1:
            raise ValueError("Phoneme token IDs must follow the text padding row (and any edit marker rows)")
        if (
            any(len(s) != 1 for s in symbols)
            or len(set(symbols)) != len(symbols)
            or {phonemes["open_marker"], phonemes["close_marker"]} & set(symbols)
        ):
            raise ValueError("Invalid phoneme symbols")
    first, n_ph = _block_first(spec), phoneme_token_count(spec)
    start = spec.get("start")
    if start is not None:
        if start["marker"] != START_MARKER:
            raise ValueError("Invalid start marker")
        if first < text_vocab + 1 or start["id"] != first + n_ph:
            raise ValueError("Start token ID must follow the phoneme rows")
    pauses = spec.get("pauses")
    if pauses is not None:
        ids = pauses["ids"]
        if (
            not 1 <= len(ids) <= 8
            or list(pauses["markers"]) != [chr(0xE003 + j) for j in range(len(ids))]
            or len(pauses["seconds"]) != len(ids)
        ):
            raise ValueError("Invalid pause markers")
        lo = first + n_ph + int(start is not None)
        if first < text_vocab + 1 or list(ids) != list(range(lo, lo + len(ids))):
            raise ValueError("Pause token IDs must follow the start row")


def phoneme_token_count(spec):
    """Text rows a spec's phoneme tokens take (0 without phonemes)."""
    phonemes = spec.get("phonemes") if spec else None
    return 0 if phonemes is None else 2 + len(phonemes["symbols"])


def extra_token_count(spec):
    """Text rows a spec's phoneme, start and pause tokens take together (0 without any)."""
    if not spec:
        return 0
    pauses = spec.get("pauses")
    return (
        phoneme_token_count(spec) + int(spec.get("start") is not None) + (0 if pauses is None else len(pauses["ids"]))
    )


def _block_first(spec):
    """First id of the phoneme / start / pause rows (None without any)."""
    if not spec:
        return None
    if spec.get("phonemes") is not None:
        return int(spec["phonemes"]["open_id"])
    if spec.get("start") is not None:
        return int(spec["start"]["id"])
    if spec.get("pauses") is not None:
        return int(spec["pauses"]["ids"][0])
    return None


def rows_before_extra_tokens(spec, text_vocab):
    """Extra text rows between the padding row and the phoneme / start / pause rows (edit marker rows)."""
    first = _block_first(spec)
    return 0 if first is None else first - int(text_vocab) - 1


def _bpe(text, spec):
    return [
        i + 4
        for i in tokenizer(spec["path"], spec["sha256"], spec["vocab_size"]).encode(text, add_special_tokens=False).ids
    ]


def payload_ids(text, spec):
    if START_MARKER in text:
        start = spec.get("start")
        if start is None:
            raise ValueError(NO_START_TOKEN)
        if not text.startswith(START_MARKER) or START_MARKER in text[1:]:
            raise ValueError(START_NOT_FIRST)
        return [start["id"]] + payload_ids(text[1:], spec)
    if PAUSE_MARKER_RE.search(text):
        pauses = spec.get("pauses")
        if pauses is None:
            raise ValueError(NO_PAUSE_TOKENS)
        ids_of = dict(zip(pauses["markers"], pauses["ids"]))
        ids, pos = [], 0
        for match in PAUSE_MARKER_RE.finditer(text):
            marker = match.group()
            if marker not in ids_of:
                raise ValueError(f"Pause marker U+{ord(marker):04X} outside the {len(ids_of)} pause buckets")
            if match.start() > pos:
                ids.extend(payload_ids(text[pos : match.start()], spec))
            ids.append(ids_of[marker])
            pos = match.end()
        if pos < len(text):
            ids.extend(payload_ids(text[pos:], spec))
        return ids
    phonemes = spec.get("phonemes")
    if phonemes is None:
        return _bpe(text, spec)
    open_, close = phonemes["open_marker"], phonemes["close_marker"]
    if open_ not in text and close not in text:
        return _bpe(text, spec)
    phone_ids = {s: phonemes["open_id"] + 2 + i for i, s in enumerate(phonemes["symbols"])}
    ids, pos = [], 0
    while pos < len(text):
        start = text.find(open_, pos)
        stretch = text[pos:] if start < 0 else text[pos:start]
        if close in stretch:
            raise ValueError("Unbalanced phoneme group markers")
        if stretch:
            ids.extend(_bpe(stretch, spec))
        if start < 0:
            break
        end = text.find(close, start + 1)
        group = text[start + 1 : end] if end >= 0 else ""
        if end < 0 or not group or open_ in group:
            raise ValueError("Unbalanced or empty phoneme group")
        unknown = sorted(set(group) - phone_ids.keys())
        if unknown:
            raise ValueError(f"Phones outside the inventory: {unknown}")
        ids.append(phonemes["open_id"])
        ids.extend(phone_ids[s] for s in group)
        ids.append(phonemes["close_id"])
        pos = end + 1
    return ids


def text_ids(text, spec):
    return [2] + payload_ids(text, spec) + [3]


def load_spec(model_path):
    from zonos3.utils.release import resolve_dir_or_local, resolve_tokenizer_path

    model_dir = resolve_dir_or_local(str(model_path))
    model = json.loads((Path(model_dir) / "params.json").read_text())["model"]
    spec = model.get("text_tokenizer_spec")
    if spec is not None:
        spec = dict(spec, path=resolve_tokenizer_path(spec["path"], model_dir))
        validate_spec(spec, model["text_vocab"])
        count = extra_token_count(spec)
        if count and rows_before_extra_tokens(spec, model["text_vocab"]) + count != int(
            model.get("text_extra_tokens", 0)
        ):
            raise ValueError(
                "Phoneme / start / pause tokens do not match text_extra_tokens (they must be the last extra rows)"
            )
    return spec
