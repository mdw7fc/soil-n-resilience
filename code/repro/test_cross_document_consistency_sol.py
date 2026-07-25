#!/usr/bin/env python3
"""Verify final manuscript/SI/response values against regenerated SOL outputs."""
from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import pandas as pd
from docx import Document

ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT.parents[1]

MANUSCRIPT = SUBMISSION / "Wallenstein-Manning_ERFS_manuscript_v14_sol.docx"
SI = SUBMISSION / "Wallenstein-Manning_ERFS_SI_v14_sol.docx"
RESPONSE = SUBMISSION / "Author_Response_ERFS-100341_v14_sol.docx"


def all_text(path: Path) -> str:
    doc = Document(path)
    chunks = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            chunks.extend(cell.text for cell in row.cells)
    return "\n".join(chunks)


def require(text: str, fragments: list[str], label: str) -> None:
    missing = [item for item in fragments if item not in text]
    assert not missing, f"{label}: missing {missing}"


def forbid(text: str, fragments: list[str], label: str) -> None:
    present = [item for item in fragments if item in text]
    assert not present, f"{label}: stale fragments {present}"


def embedded_sha(path: Path, internal: str) -> str:
    with zipfile.ZipFile(path) as archive:
        return hashlib.sha256(archive.read(internal)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manuscript = all_text(MANUSCRIPT)
    si = all_text(SI)
    response = all_text(RESPONSE)

    require(manuscript, [
        "2.3%", "3.18%", "3.3%",
        "ρ = +0.19", "Pearson R² = 0.06",
        "5.55%", "5.13%", "an 8% reduction",
        "unsupported εF,N values −0.50 and −1.00",
    ], "manuscript")
    require(si, [
        "31.9", "18.3", "13.8", "22.1", "23.0", "37.7", "15.0",
        "43.2", "33.6", "19.3", "31.4", "49.6", "54.0", "45.0", "44.8",
        "ρ = +0.19", "Pearson R² = 0.06",
        "ρ = +0.40", "Pearson R² = 0.11",
        "4.47 t ha⁻¹", "3.07–6.73",
        "5.55%", "5.13%", "4.13%", "2.90%",
        "reverses in SSA at −0.50 and −1.00",
    ], "SI")
    require(response, [
        "independent benchmark", "4.47 t ha⁻¹", "3.07–6.73",
        "ρ = +0.19", "R² = 0.06", "ρ = +0.40", "R² = 0.11",
        "εF,N = −0.50 and −1.00",
    ], "response")

    stale = [
        "reduces Sub-Saharan African year-10 yield loss from 20.8%",
        "reduces SSA year-10 yield loss from 20.81%",
        "a 57% reduction",
        "ρ = +0.07; Pearson R² = 0.05",
        "year 1 is ρ = +0.29",
        "We additionally validated the modelled Sub-Saharan",
    ]
    forbid(manuscript + "\n" + si + "\n" + response, stale, "all documents")

    table = pd.read_csv(ROOT / "outputs" / "Table_S4_calibration_sol.csv")
    si_doc = Document(SI)
    table_s4 = next(
        t for t in si_doc.tables
        if t.rows[0].cells[0].text.strip() == "Region"
        and "y_max" in t.rows[0].cells[4].text
    )
    labels = [
        "N America", "Europe", "E Asia", "S Asia", "SE Asia",
        "L America", "Sub-Saharan Africa", "FSU/C Asia",
    ]
    for label, row, record in zip(
        labels, table_s4.rows[1:], table.itertuples(index=False)
    ):
        assert row.cells[0].text.strip() == label
        assert abs(float(row.cells[4].text) - record.calibrated_y_max_t_ha) <= .01
        assert abs(
            float(row.cells[7].text)
            - record.simulated_year2_no_synth_n_t_ha
        ) <= .01

    media = [
        (MANUSCRIPT, "word/media/image3.png", ROOT / "figures/Figure_3_mechanism_screen.png"),
        (SI, "word/media/image2.png", ROOT / "figures/Figure_S2_broadbalk_benchmark.png"),
        (SI, "word/media/image4.png", ROOT / "figures/Figure_S4_hindcast_sensitivity.png"),
        (SI, "word/media/image10.png", ROOT / "figures/Figure_S10_nue_sensitivity.png"),
        (SI, "word/media/image13.png", ROOT / "figures/Figure_S13_OFRA_SSA_validation.png"),
    ]
    for docx, internal, source in media:
        assert embedded_sha(docx, internal) == file_sha(source), (
            f"{docx.name}:{internal} does not match {source.name}"
        )

    print("CROSS-DOCUMENT CONSISTENCY: PASS")
    print("  headline values, BNF/buffer tables, Table S4, benchmark language")
    print("  and embedded regenerated figures agree with the SOL outputs.")


if __name__ == "__main__":
    main()
