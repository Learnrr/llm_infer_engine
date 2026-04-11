import argparse
import signal
import time

import cpp_engine


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cpp_engine worker process")
    parser.add_argument("--config", required=True, help="Path to worker llm_engine_config json")
    args = parser.parse_args()

    running = True

    def _stop_handler(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    engine = cpp_engine.Engine.get_instance()
    engine.init(args.config)
    engine.run()

    while running:
        time.sleep(1.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
