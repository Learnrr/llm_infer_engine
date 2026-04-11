import argparse
import os
import signal
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import cpp_engine  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"Failed to import cpp_engine. Ensure cpp_engine is built under {ROOT_DIR} and PYTHONPATH includes it."
    ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PD scheduler process")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to scheduler llm_engine_config json (prefill/decode)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Keep relative paths in config (for example model_config_path) stable.
    os.chdir(config_path.parent)

    running = True

    def _stop_handler(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    engine = cpp_engine.Engine.get_instance()
    engine.init(config_path.name)
    engine.run()

    while running:
        time.sleep(1.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
