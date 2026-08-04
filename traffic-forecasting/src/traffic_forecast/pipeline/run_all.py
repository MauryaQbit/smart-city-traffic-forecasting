"""Run all stages end-to-end. Replaces the legacy train_models.py."""

from __future__ import annotations

from traffic_forecast.pipeline import stage1, stage2, stage3, stage4
from traffic_forecast.pipeline._common import ensure_dirs, load_and_split


def main(epochs: int | None = None) -> None:
    ensure_dirs()
    load_and_split()
    stage1.main()
    stage2.main()
    stage3.main(epochs=epochs)
    stage4.main()
    print("All stages complete.")


if __name__ == "__main__":
    main()
