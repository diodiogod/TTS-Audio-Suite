from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import jieba
import torch
from pypinyin import Style, lazy_pinyin
from torch.nn.utils.rnn import pad_sequence


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def is_package_available(package_name: str) -> bool:
    try:
        import importlib

        package_exists = importlib.util.find_spec(package_name) is not None
        return package_exists
    except Exception:
        return False


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt or tokenizer.json you want to use
                - "hf_tokenizer" for HuggingFace tokenizer.json files
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
                - if use "hf_tokenizer", derived from tokenizer.json vocab_size
    """
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files(__package__ or "f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        # Check if it's a tokenizer.json file or vocab.txt file
        if dataset_name.endswith(".json") and "tokenizer" in dataset_name.lower():
            # HuggingFace tokenizer.json format
            try:
                from tokenizers import Tokenizer
                hf_tokenizer = Tokenizer.from_file(dataset_name)
                vocab_char_map = hf_tokenizer.get_vocab()
                vocab_size = hf_tokenizer.get_vocab_size()

                # For F5-TTS compatibility, we need to ensure space is index 0
                if " " not in vocab_char_map or vocab_char_map[" "] != 0:
                    print(f"⚠️ Warning: tokenizer.json doesn't have space at index 0 (found at {vocab_char_map.get(' ', 'not found')})")
                    print(f"   This may cause issues with F5-TTS inference")

                print(f"🔤 Using HuggingFace tokenizer: {os.path.basename(dataset_name)} ({vocab_size} tokens)")
                return vocab_char_map, vocab_size

            except ImportError:
                print(f"❌ HuggingFace tokenizers not available, falling back to vocab.txt format")
            except Exception as e:
                print(f"❌ Failed to load tokenizer.json: {e}, falling back to vocab.txt format")

        # Standard vocab.txt format (fallback or explicit .txt file)
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    elif tokenizer == "hf_tokenizer":
        # Explicit HuggingFace tokenizer mode
        try:
            from tokenizers import Tokenizer
            hf_tokenizer = Tokenizer.from_file(dataset_name)
            vocab_char_map = hf_tokenizer.get_vocab()
            vocab_size = hf_tokenizer.get_vocab_size()

            # For F5-TTS compatibility, we need to ensure space is index 0
            if " " not in vocab_char_map or vocab_char_map[" "] != 0:
                print(f"⚠️ Warning: tokenizer.json doesn't have space at index 0 (found at {vocab_char_map.get(' ', 'not found')})")
                print(f"   This may cause issues with F5-TTS inference")

            print(f"🔤 Using HuggingFace tokenizer: {os.path.basename(dataset_name)} ({vocab_size} tokens)")

        except ImportError:
            raise ImportError("HuggingFace tokenizers package is required for tokenizer.json support. Install with: pip install tokenizers")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer.json from {dataset_name}: {e}")

    return vocab_char_map, vocab_size


# convert char to pinyin


def convert_text_with_tokenizer(text_list, vocab_char_map=None, tokenizer_path=None):
    """
    Convert text using appropriate tokenizer (HuggingFace tokenizer.json or character-based vocab.txt)

    Args:
        text_list: List of text strings to process
        vocab_char_map: Character mapping from get_tokenizer (for vocab.txt)
        tokenizer_path: Path to tokenizer.json file (for HuggingFace tokenizers)

    Returns:
        List of tokenized sequences compatible with F5-TTS
    """
    # Check if we should use HuggingFace tokenizer
    if tokenizer_path and tokenizer_path.endswith(".json") and "tokenizer" in tokenizer_path.lower():
        try:
            from tokenizers import Tokenizer
            hf_tokenizer = Tokenizer.from_file(tokenizer_path)

            # Process each text with HuggingFace tokenizer
            final_text_list = []
            for text in text_list:
                # Encode text to tokens
                encoding = hf_tokenizer.encode(text)
                # Get token strings (not IDs) - F5-TTS needs token strings
                token_strings = encoding.tokens
                # Join tokens with spaces for F5-TTS compatibility
                final_text_list.append(' '.join(token_strings))

            print(f"🔤 Processed {len(text_list)} texts with HuggingFace tokenizer")
            return final_text_list

        except ImportError:
            print(f"❌ HuggingFace tokenizers not available, falling back to character processing")
        except Exception as e:
            print(f"❌ HuggingFace tokenizer failed: {e}, falling back to character processing")

    # Fall back to standard character-based processing (for vocab.txt)
    return convert_char_to_pinyin(text_list)


def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


# get the empirically pruned step for sampling


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)
