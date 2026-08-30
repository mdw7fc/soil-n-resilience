#!/usr/bin/env python3
"""Apply the prospective 95% robustness threshold to the deposited MC draws."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "data/mc_ensemble/mc_posterior.csv.gz"
THRESHOLD = 0.95
REGIONS = [
    "north_america", "europe", "east_asia", "south_asia",
    "southeast_asia", "latin_america", "sub_saharan_africa",
    "fsu_central_asia",
]


def main():
    frame = pd.read_csv(DATA)
    assert frame["draw"].nunique() == 1000
    pivot = frame.pivot_table(
        index=["draw", "region"], columns="soc_pct", values="yield_pen"
    )
    pivot["buffer_positive"] = pivot[50] >= pivot[150]
    rates = pivot.groupby("region")["buffer_positive"].mean()
    assert set(rates.index) == set(REGIONS)
    assert (rates >= THRESHOLD).all(), rates

    by_draw = pivot["buffer_positive"].unstack("region")
    simultaneous = float(by_draw.all(axis=1).mean())
    assert simultaneous >= THRESHOLD

    print("MC ROBUSTNESS THRESHOLD: PASS")
    for key, value in rates.items():
        print(f"  {key:22s} positive-buffer frequency {value:.3f}")
    print(f"  simultaneous all-region frequency {simultaneous:.3f}")


if __name__ == "__main__":
    main()
