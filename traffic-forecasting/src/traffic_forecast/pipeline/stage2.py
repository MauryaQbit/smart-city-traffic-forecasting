"""Stage 2: SARIMA per junction on rolling 2-month window."""

from __future__ import annotations

from traffic_forecast import config
from traffic_forecast.models import sarima as sarima_model
from traffic_forecast.pipeline._common import (
    ensure_dirs,
    load_pickled_split,
    write_json,
)


def main() -> dict:
    ensure_dirs()
    train_df, val_df = load_pickled_split()
    out = sarima_model.train(train_df, val_df)
    write_json(config.REPORTS_DIR / "sarima_predictions.json", out["predictions"])
    result = {
        "SARIMA": {
            **out["metrics"],
            "note": (
                f"fit on last {config.SARIMA_TRAIN_WINDOW}h/junction, "
                f"evaluated on first {config.SARIMA_EVAL_HOURS}h of validation window"
            ),
        }
    }
    write_json(config.REPORTS_DIR / "results_stage2_sarima.json", result)
    print("Stage 2 done.")
    return result


if __name__ == "__main__":
    main()
