"""Download SoilGrids 2.0 5 km aggregated layers needed for country-level
buffer proxy.

Layers pulled (units per ISRIC SoilGrids 2.0 documentation):
  - nitrogen 0-5, 5-15, 15-30 cm (mean)            [cg/kg]      → ÷100 → g/kg
  - bdod      0-5, 5-15, 15-30 cm (mean)           [cg/cm³]     → ÷100 → g/cm³
  - ocs       0-30 cm (mean) — SOC stock           [t/ha × 10]  → ÷10  → t/ha

We use OCS for the SOC-stock value and compute total-N stock 0-30 cm by
integrating concentration × bulk density × depth slab thickness.

Total source size: ~60 MB.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from _config import DIR_RAW  # noqa: E402

BASE = "https://files.isric.org/soilgrids/latest/data_aggregated/5000m"

LAYERS = [
    # (var, depth, filename suffix)
    ("nitrogen", "0-5cm"),
    ("nitrogen", "5-15cm"),
    ("nitrogen", "15-30cm"),
    ("bdod",     "0-5cm"),
    ("bdod",     "5-15cm"),
    ("bdod",     "15-30cm"),
    ("ocs",      "0-30cm"),
    ("soc",      "0-5cm"),     # for cross-check: SOC concentration at surface
    ("soc",      "5-15cm"),
    ("soc",      "15-30cm"),
]


def url_for(var: str, depth: str) -> str:
    return f"{BASE}/{var}/{var}_{depth}_mean_5000.tif"


def local_for(var: str, depth: str) -> Path:
    return DIR_RAW / f"soilgrids_{var}_{depth}_mean_5000.tif"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name}  ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  [get ] {dest.name}")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"  [done] {dest.name}  {dest.stat().st_size/1e6:.1f} MB  "
          f"({time.time()-t0:.1f}s)")


def main() -> int:
    print("Downloading SoilGrids 5 km layers ...")
    for var, depth in LAYERS:
        download(url_for(var, depth), local_for(var, depth))
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
