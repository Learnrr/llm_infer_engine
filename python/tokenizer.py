from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from pathlib import Path

class Tokenizer:
    def __init__(self):
        self.tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))

    def load_from_path(self, tokenizer_path):
        path = Path(tokenizer_path)
        if not path.exists():
            raise FileNotFoundError(f"tokenizer file not found: {path}")
        self.tokenizer = HFTokenizer.from_file(str(path))

    def encode(self, prompt):
        if self.tokenizer.get_vocab_size() == 0:
            raise RuntimeError(
                "Tokenizer vocabulary is empty. Please load a tokenizer.json before inference."
            )
        return self.tokenizer.encode(prompt, add_special_tokens=False).ids

    def tokenize(self, prompt):
        return self.encode(prompt)

    def decode(self, token_ids):
        if self.tokenizer.get_vocab_size() == 0:
            raise RuntimeError(
                "Tokenizer vocabulary is empty. Please load a tokenizer.json before inference."
            )
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def detokenize(self, token_ids):
        return self.decode(token_ids)