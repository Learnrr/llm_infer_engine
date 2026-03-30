from python.openai_obj import (
    ChatCompletion,
    Completion,
    ChatCompletionResponse,
    CompletionResponse,
    Message,
    Choice
)
from python.request import APIRequest
from python.RequestOutput import RequestOutput
import time
import uuid


def _build_prompt_from_messages(messages: list[Message]) -> str:
    # Qwen chat template (ChatML style).
    chunks = []
    for msg in messages:
        role = (msg.role or "user").strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        content = msg.content or ""
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    chunks.append("<|im_start|>assistant\n")
    return "".join(chunks)


def chat_completion_to_request(completion: ChatCompletion) -> APIRequest:
    prompt = _build_prompt_from_messages(completion.messages)
    return APIRequest(
        prompt=prompt,
        model=completion.model,
        temperature=completion.temperature,
        top_p=completion.top_p,
        max_tokens=completion.max_tokens,
        presence_penalty=completion.presence_penalty,
        frequency_penalty=completion.frequency_penalty,
        user=completion.user
    )

def completion_to_request(completion: Completion) -> APIRequest:
    return APIRequest(
        prompt=completion.prompt,
        model=completion.model,
        temperature=completion.temperature,
        top_p=completion.top_p,
        max_tokens=completion.max_tokens,
        presence_penalty=completion.presence_penalty,
        frequency_penalty=completion.frequency_penalty,
        user=completion.user
    )

def request_to_chat_completion_response(
        api_request: APIRequest, output: RequestOutput) -> ChatCompletionResponse:
    total_tokens = len(output.token_ids)
    generated_tokens = len(output.generated_tokens)
    prompt_tokens = total_tokens - generated_tokens

    return ChatCompletionResponse(
        id=f"chatcompletion-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=api_request.model,
        choices=[
            Choice(
                message=Message(
                    role="assistant",
                    content=output.output_text
                ),
                finish_reason=total_tokens >= api_request.max_tokens and "length" or "stop",
                index=0
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated_tokens,
            "total_tokens": total_tokens
        }
    )

def request_to_completion_response(
        api_request: APIRequest, output: RequestOutput) -> CompletionResponse:
    total_tokens = len(output.token_ids)
    generated_tokens = len(output.generated_tokens)
    prompt_tokens = total_tokens - generated_tokens

    return CompletionResponse(
        id=f"completion-{uuid.uuid4().hex}",
        object="completion",
        created=int(time.time()),
        model=api_request.model,
        choices=[
            Choice(
                message=Message(
                    role="assistant",
                    content=output.output_text
                ),
                finish_reason=total_tokens >= api_request.max_tokens and "length" or "stop",
                index=0
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated_tokens,
            "total_tokens": total_tokens
        }
    )