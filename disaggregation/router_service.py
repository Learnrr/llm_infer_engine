import argparse
import importlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from python.adaptor import (  # noqa: E402
    chat_completion_to_request,
    completion_to_request,
    request_to_chat_completion_response,
    request_to_completion_response,
)
from python.engine import Engine  # noqa: E402
from python.openai_obj import (  # noqa: E402
    ChatCompletion,
    ChatCompletionResponse,
    Completion,
    CompletionResponse,
)


def create_app(config_path: str):
    fastapi_module = importlib.import_module("fastapi")
    starlette_concurrency = importlib.import_module("starlette.concurrency")
    FastAPI = fastapi_module.FastAPI
    run_in_threadpool = starlette_concurrency.run_in_threadpool

    app = FastAPI(title="PD Router Service")
    engine = Engine(config_path)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletion) -> ChatCompletionResponse:
        api_request_obj = chat_completion_to_request(request)
        request_id = await run_in_threadpool(engine.submit, api_request_obj.prompt, api_request_obj)
        output = await run_in_threadpool(engine.get_output, request_id)
        return request_to_chat_completion_response(api_request_obj, output)

    @app.post("/v1/completions")
    async def create_completion(request: Completion) -> CompletionResponse:
        api_request_obj = completion_to_request(request)
        request_id = await run_in_threadpool(engine.submit, api_request_obj.prompt, api_request_obj)
        output = await run_in_threadpool(engine.get_output, request_id)
        return request_to_completion_response(api_request_obj, output)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PD router service with FastAPI")
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "disaggregation" / "llm_engine_config_router.json"),
        help="Path to router llm_engine_config json",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()

    try:
        uvicorn = importlib.import_module("uvicorn")
    except ModuleNotFoundError as exc:
        missing = exc.name or "uvicorn"
        raise ModuleNotFoundError(
            f"Missing dependency '{missing}'. Install fastapi, uvicorn, and starlette first."
        ) from exc

    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
