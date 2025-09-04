"""
MTLTokenizer - Multilingual Tokenizer for ChatterBox Official 23-Lang
Based on ResembleAI's official multilingual tokenizer implementation
"""

import logging
import json
import re
from pathlib import Path
from unicodedata import category

import torch
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

# Model repository
REPO_ID = "ResembleAI/chatterbox"

# Global instances for optional dependencies
_kakasi = None
_dicta = None


def is_kanji(c: str) -> bool:
    """Check if character is kanji."""
    return 19968 <= ord(c) <= 40959


def is_katakana(c: str) -> bool:
    """Check if character is katakana."""
    return 12449 <= ord(c) <= 12538


def hiragana_normalize(text: str) -> str:
    """Japanese text normalization: converts kanji to hiragana; katakana remains the same."""
    global _kakasi
    
    try:
        if _kakasi is None:
            import pykakasi
            _kakasi = pykakasi.kakasi()
        
        result = _kakasi.convert(text)
        out = []
        
        for r in result:
            inp = r['orig']
            hira = r["hira"]

            # Any kanji in the phrase
            if any([is_kanji(c) for c in inp]):
                if hira and hira[0] in ["は", "へ"]:  # Safety check for empty hira
                    hira = " " + hira
                out.append(hira)

            # All katakana
            elif all([is_katakana(c) for c in inp]) if inp else False:  # Safety check for empty inp
                out.append(r['orig'])

            else:
                out.append(inp)
        
        normalized_text = "".join(out)
        
        # Decompose Japanese characters for tokenizer compatibility
        import unicodedata
        normalized_text = unicodedata.normalize('NFKD', normalized_text)
        
        return normalized_text
        
    except ImportError:
        logger.warning("pykakasi not available - Japanese text processing skipped")
        return text


def add_hebrew_diacritics(text: str) -> str:
    """Hebrew text normalization: adds diacritics to Hebrew text."""
    global _dicta
    
    try:
        if _dicta is None:
            from dicta_onnx import Dicta
            _dicta = Dicta()
        
        return _dicta.add_diacritics(text)
        
    except ImportError:
        logger.warning("dicta_onnx not available - Hebrew text processing skipped")
        return text
    except Exception as e:
        logger.warning(f"Hebrew diacritization failed: {e}")
        return text


def korean_normalize(text: str) -> str:
    """Korean text normalization: decompose syllables into Jamo for tokenization."""
    
    def decompose_hangul(char):
        """Decompose Korean syllable into Jamo components."""
        if not ('\uac00' <= char <= '\ud7af'):
            return char
        
        # Hangul decomposition formula
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ''
        
        return initial + medial + final
    
    # Decompose syllables and normalize punctuation
    result = ''.join(decompose_hangul(char) for char in text)    
    return result.strip()


class ChineseCangjieConverter:
    """Converts Chinese characters to Cangjie codes for tokenization."""
    
    def __init__(self, model_dir=None):
        self.word2cj = {}
        self.cj2word = {}
        self.segmenter = None
        self._load_cangjie_mapping(model_dir)
        self._init_segmenter()
    
    def _load_cangjie_mapping(self, model_dir=None):
        """Load Cangjie mapping from HuggingFace model repository."""        
        try:
            cangjie_file = hf_hub_download(
                repo_id=REPO_ID,
                filename="Cangjie5_TC.json",
                cache_dir=model_dir
            )
            
            with open(cangjie_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            
            for entry in data:
                word, code = entry.split("\t")[:2]
                self.word2cj[word] = code
                if code not in self.cj2word:
                    self.cj2word[code] = [word]
                else:
                    self.cj2word[code].append(word)
                    
        except Exception as e:
            logger.warning(f"Could not load Cangjie mapping: {e}")
    
    def _init_segmenter(self):
        """Initialize pkuseg segmenter."""
        try:
            from pkuseg import pkuseg
            self.segmenter = pkuseg()
        except ImportError:
            logger.warning("pkuseg not available - Chinese segmentation will be skipped")
            self.segmenter = None
    
    def _cangjie_encode(self, glyph: str):
        """Encode a single Chinese glyph to Cangjie code."""
        normed_glyph = glyph
        code = self.word2cj.get(normed_glyph, None)
        if code is None:  # e.g. Japanese hiragana
            return None
        index = self.cj2word[code].index(normed_glyph)
        index = str(index) if index > 0 else ""
        return code + str(index)
    
    def __call__(self, text):
        """Convert Chinese characters in text to Cangjie tokens."""
        output = []
        if self.segmenter is not None:
            segmented_words = self.segmenter.cut(text)
            full_text = " ".join(segmented_words)
        else:
            full_text = text
        
        for char in full_text:
            if '\u4e00' <= char <= '\u9fff':  # CJK Unified Ideographs
                cangjie_code = self._cangjie_encode(char)
                if cangjie_code:
                    output.append(cangjie_code)
                else:
                    output.append(char)
            else:
                output.append(char)
        
        return ''.join(output)


class MTLTokenizer:
    """Multilingual tokenizer for ChatterBox Official 23-Lang"""
    
    def __init__(self, vocab_file_path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        model_dir = Path(vocab_file_path).parent
        self.cangjie_converter = ChineseCangjieConverter(model_dir)
        self.check_vocabset_sot_eot()
    
    def check_vocabset_sot_eot(self):
        """Verify that required special tokens are in vocabulary"""
        voc = self.tokenizer.get_vocab()
        assert SOT in voc, f"Special token {SOT} not found in vocabulary"
        assert EOT in voc, f"Special token {EOT} not found in vocabulary"
    
    def text_to_tokens(self, text: str, language_id: str = None):
        """Convert text to token tensor"""
        text_tokens = self.encode(text, language_id=language_id)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens
    
    def encode(self, txt: str, language_id: str = None):
        """
        Encode text to tokens with language-specific preprocessing
        
        Args:
            txt: Input text
            language_id: Language identifier (ar, da, de, etc.)
            
        Returns:
            List of token IDs
        """
        # Language-specific text processing
        if language_id == 'zh':
            txt = self.cangjie_converter(txt)
        elif language_id == 'ja':
            txt = hiragana_normalize(txt)
        elif language_id == 'he':
            txt = add_hebrew_diacritics(txt)
        elif language_id == 'ko':
            txt = korean_normalize(txt)
        
        # Prepend language token if specified
        if language_id:
            txt = f"[{language_id.lower()}]{txt}"
        
        # Replace spaces with SPACE token
        txt = txt.replace(' ', SPACE)
        
        # Encode using the tokenizer
        return self.tokenizer.encode(txt).ids
    
    def decode(self, seq):
        """Decode token sequence back to text"""
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        
        txt = self.tokenizer.decode(seq, skip_special_tokens=False)
        # Clean up the decoded text
        txt = txt.replace(' ', '').replace(SPACE, ' ').replace(EOT, '').replace(UNK, '')
        
        # Remove language prefixes like [en], [fr], etc.
        txt = re.sub(r'^\[[a-z]{2}\]', '', txt)
        
        return txt
    
    def get_vocab_size(self):
        """Get vocabulary size"""
        return self.tokenizer.get_vocab_size()
    
    def get_vocab(self):
        """Get full vocabulary"""
        return self.tokenizer.get_vocab()


# Legacy compatibility class - redirect to MTLTokenizer
class EnTokenizer(MTLTokenizer):
    """Legacy English tokenizer - now redirects to MTLTokenizer for compatibility"""
    
    def __init__(self, vocab_file_path):
        # If this is the old tokenizer format, we need to handle it differently
        if vocab_file_path.endswith('tokenizer.json') and not vocab_file_path.endswith('mtl_tokenizer.json'):
            # This is likely the old English-only tokenizer - use original implementation
            logger.warning("Using legacy English tokenizer - multilingual features not available")
            self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
            self.check_vocabset_sot_eot()
            self.is_legacy = True
        else:
            # This is the new multilingual tokenizer
            super().__init__(vocab_file_path)
            self.is_legacy = False
    
    def check_vocabset_sot_eot(self):
        """Verify that required special tokens are in vocabulary"""
        voc = self.tokenizer.get_vocab()
        assert SOT in voc, f"Special token {SOT} not found in vocabulary"
        assert EOT in voc, f"Special token {EOT} not found in vocabulary"
    
    def text_to_tokens(self, text: str):
        """Legacy interface - assumes English"""
        if self.is_legacy:
            # Use the old English-only behavior
            text_tokens = self.encode(text)
            text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
            return text_tokens
        else:
            # Use multilingual tokenizer with English as default
            return super().text_to_tokens(text, language_id="en")
    
    def encode(self, txt: str, verbose=False):
        """Legacy interface - assumes English"""
        if self.is_legacy:
            # Use the old English-only behavior
            txt = txt.replace(' ', SPACE)
            code = self.tokenizer.encode(txt)
            return code.ids
        else:
            # Use multilingual tokenizer with English as default
            return super().encode(txt, language_id="en")
    
    def decode(self, seq):
        """Legacy decode method"""
        if self.is_legacy:
            # Use original implementation
            if isinstance(seq, torch.Tensor):
                seq = seq.cpu().numpy()

            txt: str = self.tokenizer.decode(seq, skip_special_tokens=False)
            txt = txt.replace(' ', '')
            txt = txt.replace(SPACE, ' ')
            txt = txt.replace(EOT, '')
            txt = txt.replace(UNK, '')
            return txt
        else:
            # Use multilingual implementation
            return super().decode(seq)