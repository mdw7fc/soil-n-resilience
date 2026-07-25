"""Download Natural Earth admin-0 (countries) at 1:50m.

Natural Earth Public Domain — direct ZIP from https://naturalearth.s3.amazonaws.com/.
Saved as a shapefile bundle so geopandas can read it without network access.
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from _config import DIR_RAW  # noqa: E402

URL = ("https://naturalearth.s3.amazonaws.com/"
       "50m_cultural/ne_50m_admin_0_countries.zip")
ZIP_PATH = DIR_RAW / "ne_50m_admin_0_countries.zip"
EXTRACT_DIR = DIR_RAW / "ne_50m_admin_0_countries"


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"  [get ] {URL}")
        t0 = time.time()
        with requests.get(URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with ZIP_PATH.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"  [done] {ZIP_PATH.name}  "
              f"{ZIP_PATH.stat().st_size/1e6:.1f} MB  "
              f"({time.time()-t0:.1f}s)")
    else:
        print(f"  [skip] {ZIP_PATH.name} already present")

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"  extracted to {EXTRACT_DIR}")
    print(f"  files: {sorted(p.name for p in EXTRACT_DIR.iterdir())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
