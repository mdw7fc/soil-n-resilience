#!/usr/bin/env python3
"""Repo documentation carries the released numbers, or says it is history.

The third external audit found the reproducibility archive's own prose stale:
README.md and MANIFEST.md still quoted the pre-F-025/F-026 result family
(global 2.31 / 3.18 / 3.29, SSA y_max 3.88, a zero central eps_F_N) while the
claim register gated the documents and never read the repo docs at all. A
"70/70 AGREES" check that scans the manuscript but not the README catches
drift in one place and ships it in another.

This test closes that hole with a three-tier rule (F-028):

  1. CURRENT docs (README.md, MANIFEST.md, HANDOFF.md) must contain the
     released headline family and must not contain any stale fragment.
  2. LEDGER docs (FINDINGS.md, CHANGELOG.md) are append-only history; old
     numbers appear there legitimately, inside dated entries. Exempt from the
     forbid scan, but CHANGELOG.md must state the current family somewhere.
  3. Every OTHER tracked markdown file at the repo root and under results/
     must either carry the string "SUPERSEDED" in its first 400 characters or
     contain no stale fragment. A working record is welcome to stay in the
     tree; it just has to say what it is.

Run: python3 code/repro/test_repo_docs_consistency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# The released headline family. These change only with a FINDINGS entry that
# authorises a refreeze of the claim register (same rule as docs/claims.yaml).
REQUIRE_README = [
    "2.32 / 3.02 / 3.07",        # global S3 loss, years 1/10/30
    "3.97",                      # SSA y_max, production-path calibration
    "-0.50",                     # central eps_F_N (F-026)
    "docs/claims.yaml",          # the authoritative register
]
REQUIRE_MANIFEST = [
    "docs/claims.yaml",
]
REQUIRE_HANDOFF = [
    "2.32 / 3.02 / 3.07",
    "-0.50",
]
REQUIRE_CHANGELOG = [
    "2.32 / 3.02 / 3.07",
]

# Fragments that identify the pre-F-025/F-026 family. Distinctive enough not
# to collide with legitimate current text; a false positive here is a prompt
# to look, which is the job of a gate.
STALE = [
    "2.31 / 3.18 / 3.29",
    "2.31%, 3.18%, and 3.29%",
    "0.32 of 3.18 pp",
    "y_max = 3.88",
    "SSA 3.88",
    "SSA y_max = 3.88",
    "eps_F_N = 0`",
    "eps_F_N` = 0",
    "central eps_F_N is zero",
    "5.57 % (FSU)",
    "+5.32 / +6.27 / +6.59",
]

CURRENT = {"README.md": REQUIRE_README, "MANIFEST.md": REQUIRE_MANIFEST,
           "HANDOFF.md": REQUIRE_HANDOFF}
LEDGER = {"FINDINGS.md", "CHANGELOG.md"}

failures: list[str] = []


def check_current(rel: str, need: list[str]) -> None:
    text = (ROOT / rel).read_text(encoding="utf-8")
    for frag in need:
        if frag not in text:
            failures.append(f"{rel}: missing required fragment {frag!r}")
    for frag in STALE:
        if frag in text:
            failures.append(f"{rel}: stale fragment {frag!r}")


def check_other(path: Path) -> None:
    rel = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    if "SUPERSEDED" in text[:400]:
        return
    for frag in STALE:
        if frag in text:
            failures.append(
                f"{rel}: stale fragment {frag!r} in a file that neither "
                f"carries a SUPERSEDED banner nor is a ledger doc")


def main() -> None:
    for rel, need in CURRENT.items():
        check_current(rel, need)

    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for frag in REQUIRE_CHANGELOG:
        if frag not in text:
            failures.append(f"CHANGELOG.md: missing required fragment {frag!r}"
                            " (add a dated entry with the released family)")

    scan: list[Path] = sorted(ROOT.glob("*.md")) + sorted(
        (ROOT / "results").glob("*.md"))
    for path in scan:
        name = path.name
        if name in CURRENT or name in LEDGER:
            continue
        check_other(path)

    if failures:
        print("REPO DOCS CONSISTENCY: FAIL")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("REPO DOCS CONSISTENCY: OK "
          f"({len(CURRENT) + 1} current/ledger docs checked, "
          f"{len(scan)} markdown files scanned)")


if __name__ == "__main__":
    main()
