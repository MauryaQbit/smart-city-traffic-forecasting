"""Entry point for `python -m traffic_forecast <subcommand>`."""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m traffic_forecast {generate-data|stage1|stage2|stage3|combine|all}")
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd == "generate-data":
        from traffic_forecast.data.generate import main as run
    elif cmd == "stage1":
        from traffic_forecast.pipeline.stage1 import main as run
    elif cmd == "stage2":
        from traffic_forecast.pipeline.stage2 import main as run
    elif cmd == "stage3":
        from traffic_forecast.pipeline.stage3 import main as run
    elif cmd == "combine":
        from traffic_forecast.pipeline.stage4 import main as run
    elif cmd == "all":
        from traffic_forecast.pipeline.run_all import main as run
    else:
        print(f"unknown subcommand: {cmd}")
        sys.exit(2)
    run()


if __name__ == "__main__":
    main()
