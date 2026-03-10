import logging

import torch
from tokenizers import Tokenizer


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

class EnTokenizer:
    def __init__(self, vocab_file_path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self.vocab = self.tokenizer.get_vocab()
        # TTS Audio Suite patch: adapt tokenizer behavior to community vocab variants
        # that use lowercase Cyrillic and "[space]" instead of the stock "[SPACE]".
        self.space_token = self._detect_space_token()
        self.lowercase_input = self._should_lowercase_input()
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        assert SOT in self.vocab
        assert EOT in self.vocab

    def _detect_space_token(self):
        lower_space_variants = [token for token in self.vocab if token.startswith("[space]")]
        if lower_space_variants:
            return "[space]"
        return SPACE

    def _should_lowercase_input(self):
        has_lower_cyrillic = any(
            any('\u0430' <= ch <= '\u044f' or ch == '\u0451' for ch in token)
            for token in self.vocab
        )
        has_upper_cyrillic = any(
            any('\u0410' <= ch <= '\u042f' or ch == '\u0401' for ch in token)
            for token in self.vocab
        )
        return has_lower_cyrillic and not has_upper_cyrillic

    def text_to_tokens(self, text: str):
        text_tokens = self.encode(text)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode( self, txt: str, verbose=False):
        """
        clean_text > (append `lang_id`) > replace SPACE > encode text using Tokenizer
        """
        if self.lowercase_input:
            txt = txt.lower()
        txt = txt.replace(' ', self.space_token)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt: str = self.tokenizer.decode(seq,
        skip_special_tokens=False)
        txt = txt.replace(' ', '')
        txt = txt.replace(SPACE, ' ')
        txt = txt.replace("[space]", ' ')
        txt = txt.replace(EOT, '')
        txt = txt.replace(UNK, '')
        return txt
