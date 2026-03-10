from tokenizers import Tokenizer
from tokenizers.models import BPE

class Tokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(BPE())
        return

    def tokenize(self, prompt):
        token_ids = []
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        return token_ids

    def detokenize(self, token_ids):
        output = ""
        self.tokenizer.decode(token_ids, skip_special_tokens=False)
        return output