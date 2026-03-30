

from python.tokenizer import Tokenizer
import cpp_engine
from python.request import Request
from python.RequestOutput import RequestOutput
import threading

class Engine:
    def __init__(self, llm_engine_config_path):
        self.tokenizer = Tokenizer()
        self.tokenizer.load_from_path("/llm_infer_engine/weights/Qwen2.5-7B-Instruct/tokenizer.json")
        self.cpp_engine = cpp_engine.Engine.get_instance()
        self.cpp_engine.init(llm_engine_config_path)
        self.cpp_engine.run()
        self.requests = {}
        self._requests_lock = threading.Lock()
        return
    

    def submit(self, prompt):
        token_ids = self.tokenizer.encode(prompt)
        request_id = self.cpp_engine.create_request(token_ids)
        request = Request(request_id, prompt)
        request.token_ids = token_ids
        with self._requests_lock:
            self.requests[request_id] = request
        self.cpp_engine.submit_request(request_id)
        return request_id
    
    def get_output(self, request_id):
        with self._requests_lock:
            if request_id not in self.requests:
                raise ValueError(f"Request ID {request_id} not found.")
            request = self.requests[request_id]

        output = self.cpp_engine.get_request_output(request_id)
        # at this point, output.token_ids contains both prompt and generated tokens,
        # request.token_ids contains only prompt tokens,
        prompt_token_len = len(request.token_ids)
        generated_token_ids = output.token_ids[prompt_token_len:]
        output_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        req_output = RequestOutput(
            request_id, 
            output.seq_id, 
            output.token_ids, 
            output_text,
            generated_token_ids
        )
        with self._requests_lock:
            self.requests.pop(request_id, None)
        return req_output

    def check_request_state(self, request_id):
        with self._requests_lock:
            if request_id not in self.requests:
                raise ValueError(f"Request ID {request_id} not found.")
        state = self.cpp_engine.check_request_state(request_id)
        return state