"""Launcher for the Streamlit dashboard.

Lives outside `app.py` so the `tf-dashboard` entry point doesn't double-spawn
Streamlit. (When Streamlit runs `app.py` as a script, `__name__ == "__main__"`
- an in-file `main()`/guard would fire recursively.)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    from traffic_forecast.dashboard import app

    app_path = Path(app.__file__)
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)


if __name__ == "__main__":
    main()
