"""
Synthetic hourly traffic-volume generator for 4 city junctions.

Schema matches the original Kaggle-style dataset referenced in the project
reports: DateTime, Junction, Vehicles, ID. Behaviour mirrors the EDA findings
- daily + weekly seasonality, junction-specific volume levels, holiday dips,
occasional genuine spikes, a small fraction of missing values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from traffic_forecast.config import END, HOLIDAYS, JUNCTIONS, MISSING_RATIO, SEED, SPIKE_RATE, START


def hourly_profile(hour: np.ndarray) -> np.ndarray:
    """Two-peak (morning + evening rush) daily shape, values in [0, ~1.5)."""
    morning = np.exp(-((hour - 9) ** 2) / (2 * 2.0**2))
    evening = np.exp(-((hour - 18.5) ** 2) / (2 * 2.5**2))
    night_floor = 0.12
    return np.clip(night_floor + 0.55 * morning + 0.75 * evening, 0, None)


def weekday_factor(dow: np.ndarray) -> np.ndarray:
    """Weekdays busier than weekends (dow: 0=Mon ... 6=Sun)."""
    return np.where(dow < 5, 1.0, 0.65)


def build_dataset(
    start: str = START,
    end: str = END,
    junctions: dict | None = None,
    seed: int = SEED,
    missing_ratio: float = MISSING_RATIO,
    spike_rate: float = SPIKE_RATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    junctions = junctions or JUNCTIONS
    dt_index = pd.date_range(start, end, freq="h")
    frames = []
    for jid, cfg in junctions.items():
        hour = dt_index.hour.values
        dow = dt_index.dayofweek.values
        month = dt_index.month.values

        daily = hourly_profile(hour)
        weekly = weekday_factor(dow)
        yearly = 1.0 - 0.10 * np.isin(month, [6, 7, 8, 9]).astype(float)

        vol = cfg["base"] * (0.4 + daily) * weekly * yearly
        vol += rng.normal(0, cfg["noise"], size=len(dt_index))

        is_holiday = np.isin(dt_index.normalize(), HOLIDAYS)
        vol = np.where(is_holiday, vol * 0.55, vol)

        spike_mask = rng.random(len(dt_index)) < spike_rate
        vol = np.where(spike_mask, vol + rng.uniform(20, 45, len(dt_index)), vol)

        vol = np.clip(vol, 0, None)
        vol = np.round(vol).astype(int)

        frames.append(
            pd.DataFrame(
                {
                    "DateTime": dt_index,
                    "Junction": jid,
                    "Vehicles": vol,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["DateTime", "Junction"]).reset_index(drop=True)
    df["ID"] = df["DateTime"].dt.strftime("%Y%m%d%H").astype(str) + df["Junction"].astype(str)

    missing_idx = rng.choice(df.index, size=int(missing_ratio * len(df)), replace=False)
    df.loc[missing_idx, "Vehicles"] = np.nan

    return df[["DateTime", "Junction", "Vehicles", "ID"]]


def main() -> None:
    out_dir = Path(__file__).resolve().parents[3] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    out_path = out_dir / "traffic_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
