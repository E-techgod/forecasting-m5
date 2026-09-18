"""Download the M5 Forecasting - Accuracy dataset from Kaggle into data/raw/.

Requires a Kaggle API token: create one at https://www.kaggle.com/settings -> API ->
"Create New Token", which downloads kaggle.json. Place it at ~/.kaggle/kaggle.json
(chmod 600), or set KAGGLE_USERNAME / KAGGLE_KEY env vars. You must also have joined
the competition at https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules
(one click, no cost) before the API will let you download.
"""

import subprocess
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw")
COMPETITION = "m5-forecasting-accuracy"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / f"{COMPETITION}.zip"

    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW_DIR)],
        check=True,
    )

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()

    print(f"M5 data extracted to {RAW_DIR}/")


if __name__ == "__main__":
    main()
