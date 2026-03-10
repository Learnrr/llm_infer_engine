

from python.tokenizer import Tokenizer
from bridge.pybind import cpp_engine
from python.sequence import Sequence

class Engine:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.sequences = {}
        self.cpp_engine = cpp_engine.Engine()
        return
    
    def add_sequence(self, sequence_id, prompt):
        if sequence_id in self.sequences:
            raise ValueError(f"Sequence ID {sequence_id} already exists.")
        
        sequence = Sequence(sequence_id, prompt)
        self.sequences[sequence_id] = sequence
        
        # Tokenize the prompt and add to the C++ engine
        token_ids = self.tokenizer.tokenize(prompt)
        self.cpp_engine.create_sequence(sequence_id, token_ids)
