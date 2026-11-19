"""Build one verifiable CSSC handoff ZIP from the composed sources."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "CSSC_SUBMISSION_20261112"
EXP = ROOT / "exp1"
DEST = ROOT / "CSSC_SUBMISSION_20261112_POLISHED.zip"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if DEST.exists():
        raise FileExistsError(DEST)
    files: list[tuple[Path, str]] = []
    for folder, allow in (
        (SUB / "Manuscript", {".tex", ".bib", ".cls", ".pdf"}),
        (SUB / "Supporting_Information", {".tex", ".csv", ".cls", ".pdf", ".txt"}),
    ):
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in allow:
                files.append((path, path.relative_to(SUB).as_posix()))

    supplemental_zip = ROOT / "WILEY_SUBMISSION_FINAL-V1.2改" / "06_Supplementary_Material_for_Review" / "SuppInfo_Source_Data.zip"
    files.append((supplemental_zip, "Reproducibility/SuppInfo_Source_Data.zip"))
    for filename in (
        "run_cssc_gaussian_baseline.py", "run_exp1_pilot.py", "experiment_core.py",
        "compose_cssc_submission.py", "compress_cssc_main.py", "polish_cssc_narrative.py",
    ):
        path = EXP / "code" / filename
        files.append((path, "Reproducibility/exp1/code/" + filename))
    for folder in (EXP / "code" / "legacy", EXP / "code" / "legacy_engine"):
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".json"}:
                files.append((path, "Reproducibility/exp1/code/" + path.relative_to(EXP / "code").as_posix()))
    for filename in ("exp1_protocol.json", "cssc_gaussian_baseline_protocol_20261112.json"):
        path = EXP / "config" / filename
        files.append((path, "Reproducibility/exp1/config/" + filename))
    for filename in ("history_registry_pilot100_parallel.csv", "decision_rows_pilot100_parallel.csv.gz"):
        path = EXP / "outputs" / "raw" / filename
        files.append((path, "Reproducibility/exp1/outputs/raw/" + filename))
    baseline = EXP / "outputs" / "cssc_gaussian_baseline_20261112.json"
    files.append((baseline, "Reproducibility/exp1/outputs/" + baseline.name))

    intro = """# CSSC manuscript handoff — 12 November 2026

This ZIP contains the 24-page double-spaced main manuscript, its LaTeX
source in the publisher's Interact class, Supporting Information,
figures and tables, and the analysis archive. The main manuscript uses
author-year references and an abstract below 200 words.

The new zero-truncated Gaussian comparator is a post hoc baseline on
frozen EXP1 histories, not a newly selected model. Its protocol, source
history registry, paired archived decision rows, code, and output are in
`Reproducibility/exp1`. The source-data ZIP is the retained archive
for the original manuscript and supplementary results. Its own README
describes which raw external high-frequency inputs are outside the archive.

To regenerate the Gaussian output in a fresh extracted copy, remove the
copied output JSON first, install Python 3.12 with numpy, pandas, scipy,
PyYAML, and the packages imported by the legacy engine, then run
`python Reproducibility/exp1/code/run_cssc_gaussian_baseline.py` from any
working directory. The script checks all 200 regenerated 60-month
history fingerprints and oracle values before writing a new output.

The manuscript's external tariff endpoint is a hypothetical charge
under the stated tariff family, not an observed customer bill. Final
submission metadata and all author approvals should be checked by the
corresponding author in the journal portal.
"""
    manifest = {name: {"bytes": path.stat().st_size, "sha256": sha(path)} for path, name in sorted(files, key=lambda x: x[1])}
    if len(manifest) != len(files):
        raise ValueError("Duplicate archive path")
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with zipfile.ZipFile(DEST, "w", allowZip64=True) as out:
        out.writestr("README.md", intro.encode("utf-8"))
        out.writestr("MANIFEST.json", manifest_bytes)
        for path, name in files:
            out.write(path, name, compress_type=zipfile.ZIP_STORED if path == supplemental_zip else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(DEST) as check:
        bad = check.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failed: {bad}")
    print(json.dumps({"zip": str(DEST), "bytes": DEST.stat().st_size, "files": len(manifest), "sha256": sha(DEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
