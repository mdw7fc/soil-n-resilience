"""Download FAOSTAT bulk CSVs for fertilizer-by-nutrient and land-use.

Outputs (data_raw/):
  - faostat_fertilizers_nutrient.zip  (Inputs / Fertilizers by Nutrient)
  - faostat_landuse.zip               (Inputs / Land Use)

Both zips contain a Normalized (long-format) CSV with columns:
  Area, Area Code, Item, Item Code, Element, Element Code, Year, Unit, Value, Flag

Bulk URLs are pinned in _config.py so re-runs are reproducible.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from _config import (  # noqa: E402
    FAOSTAT_FERT_NUTRIENT_URL,
    FAOSTAT_FERT_ZIP,
    FAOSTAT_LANDUSE_URL,
    FAOSTAT_LANDUSE_ZIP,
)


def download(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  [get ] {url}")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"  [done] {dest.name}  {dest.stat().st_size/1e6:.1f} MB  ({time.time()-t0:.1f}s)")


def main() -> int:
    print("Downloading FAOSTAT bulk CSVs ...")
    download(FAOSTAT_FERT_NUTRIENT_URL, FAOSTAT_FERT_ZIP)
    download(FAOSTAT_LANDUSE_URL, FAOSTAT_LANDUSE_ZIP)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
