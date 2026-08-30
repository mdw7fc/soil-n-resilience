"""Download MIRCA2000 cropland grids (Portmann et al. 2010) from Zenodo.

Source: Zenodo record 7422506
  - maximum_cropped_area_grid.zip   (15 MB)
      Maximum cropped area per 5 arcmin grid cell across all crops/seasons.
      Used as the "cropland-fraction" weight for soil-buffer aggregation.
  - cell_area_grid.zip               (small)
      Cell-area grid in hectares. Needed to convert cropped-area (ha) to
      cropland fraction (ha/ha).

Files are openly licensed CC-BY 4.0.
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from _config import DIR_RAW  # noqa: E402

ZENODO_ID = "7422506"
BASE = f"https://zenodo.org/api/records/{ZENODO_ID}/files"
FILES = [
    "maximum_cropped_area_grid.zip",
    "cell_area_grid.zip",
]


def download(name: str) -> Path:
    url = f"{BASE}/{name}/content"
    dest = DIR_RAW / f"mirca2000_{name}"
    if dest.exists():
        print(f"  [skip] {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    print(f"  [get ] {url}")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"  [done] {dest.name} {dest.stat().st_size/1e6:.1f} MB "
          f"({time.time()-t0:.1f}s)")
    return dest


def main() -> int:
    print("Downloading MIRCA2000 ...")
    paths = [download(n) for n in FILES]

    extract_dir = DIR_RAW / "mirca2000"
    extract_dir.mkdir(exist_ok=True)
    for p in paths:
        with zipfile.ZipFile(p) as zf:
            zf.extractall(extract_dir)
    print(f"\n  extracted to: {extract_dir}")
    files = sorted(p for p in extract_dir.iterdir() if p.is_file())
    for p in files:
        print(f"    {p.name}  ({p.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
