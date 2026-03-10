
class Sequence:
    def __init__(self, sequence_id, prompt):
        self.sequence_id = sequence_id
        self.prompt = prompt
        self.length = 0
        self.token_ids = []
