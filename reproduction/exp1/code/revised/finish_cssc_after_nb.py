"""Finish and verify the CSSC submission after the NB--Weibull run passes."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REV = ROOT / "CSSC_REVIEW_REVISION_20261112"
SUB = ROOT / "CSSC_SUBMISSION_20261112"
PORTAL = ROOT / "CSSC_PORTAL_20261115"
LOGS = REV / "postprocess_logs"
STATE = LOGS / "submission_finish_state.json"
NB = REV / "nb_weibull_cv_crps_20261115" / "run_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state(status: str, stage: str, **details: object) -> None:
    LOGS.mkdir(exist_ok=True)
    payload = {"status": status, "stage": stage,
               "time_local": time.strftime("%Y-%m-%d %H:%M:%S %z"), **details}
    temp = STATE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(STATE)


def execute(stage: str, command: list[str], cwd: Path = ROOT) -> None:
    log = LOGS / f"submission_{stage}.log"
    state("RUNNING", stage, command=command, log=str(log))
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{stage} exited {result.returncode}; inspect {log}")


def pdf_pages(path: Path) -> int:
    info = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True,
                          check=True).stdout
    match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"No page count for {path}")
    return int(match.group(1))


def audit() -> dict[str, object]:
    main_tex = SUB / "Manuscript" / "Manuscript.tex"
    main = main_tex.read_text(encoding="utf-8")
    abstract = main.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
    words = len(re.findall(r"\b[\w$-]+\b", re.sub(r"\\[A-Za-z]+", " ", abstract)))
    short = re.search(r"\\textbf\{Short title:\}\s*([^\n]+)", main)
    if short is None or len(short.group(1).strip()) > 50 or words >= 200:
        raise ValueError(f"CSSC title/abstract limit: title={short}, words={words}")
    main_pages = pdf_pages(SUB / "Manuscript" / "Manuscript.pdf")
    si_pages = pdf_pages(SUB / "Supporting_Information" / "Supporting_Information.pdf")
    if main_pages > 25:
        raise ValueError(f"Main manuscript exceeds normally expected 25 pages: {main_pages}")
    for directory, stem in ((SUB / "Manuscript", "Manuscript"),
                            (SUB / "Supporting_Information", "Supporting_Information")):
        log = (directory / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
        if re.search(r"(?:Citation|Reference) [`'][^`']+['`] .*undefined|"
                     r"There were undefined (?:references|citations)", log):
            raise ValueError(f"Unresolved LaTeX reference in {stem}.log")
        bibliography_log = directory / f"{stem}.blg"
        if (bibliography_log.is_file()
                and "I didn't find a database entry" in bibliography_log.read_text(
                    encoding="utf-8", errors="replace")):
            raise ValueError(f"Unresolved bibliography entry in {stem}.blg")
    if "for coding" in main.lower() or "for coding" in (
            SUB / "Supporting_Information" / "Supporting_Information.tex").read_text(
                encoding="utf-8").lower():
        raise ValueError("Prohibited AI-use wording remains")
    manifest = PORTAL / "CSSC_Portal_Manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Portal manifest is empty")
    names = set()
    zip_count = 0
    for row in rows:
        name = row["filename"]
        if (name in names or not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9]+", name)):
            raise ValueError(f"Invalid or duplicate upload name: {name}")
        names.add(name)
        path = PORTAL / name
        if (not path.is_file() or path.stat().st_size != int(row["bytes"])
                or path.stat().st_size >= 128_000_000 or sha256(path) != row["sha256"]):
            raise ValueError(f"Portal file integrity/size mismatch: {name}")
        if path.suffix.lower() == ".zip":
            zip_count += 1
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"ZIP integrity failure: {name}")
    if zip_count < 12:
        raise ValueError(f"Expected data, history, source and code ZIPs: {zip_count}")
    return {"status": "PASS", "main_pages": main_pages,
            "supplement_pages": si_pages, "abstract_words": words,
            "short_title_characters": len(short.group(1).strip()),
            "portal_files": len(rows), "zip_files": zip_count,
            "max_upload_bytes": max((PORTAL / row["filename"]).stat().st_size for row in rows)}


def main() -> None:
    state("WAITING", "nb_weibull", manifest=str(NB))
    while True:
        if NB.is_file():
            nb = json.loads(NB.read_text(encoding="utf-8"))
            if nb.get("status") == "FAIL":
                raise RuntimeError("NB--Weibull run failed")
            if nb.get("status") == "PASS":
                if (nb.get("chunks_passed") != 400
                        or nb.get("summary_validation", {}).get("status") != "PASS"):
                    raise ValueError("NB--Weibull completion manifest is inconsistent")
                break
        time.sleep(60)
    execute("compose", [sys.executable, str(ROOT / "exp1/code/compose_cssc_submission.py")])
    execute("main_pdf", ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                         "-file-line-error", "Manuscript.tex"], SUB / "Manuscript")
    execute("supplement_pdf", ["latexmk", "-pdf", "-interaction=nonstopmode",
                               "-halt-on-error", "-file-line-error",
                               "Supporting_Information.tex"], SUB / "Supporting_Information")
    execute("portal", [sys.executable, str(ROOT / "exp1/code/build_cssc_portal_package.py")])
    result = audit()
    report = REV / "CSSC_Final_Submission_Audit.json"
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    state("PASS", "complete", audit=str(report), portal=str(PORTAL),
          **{key: value for key, value in result.items() if key != "status"})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        state("FAIL", "submission_finish", error=repr(exc), traceback=traceback.format_exc())
        raise
