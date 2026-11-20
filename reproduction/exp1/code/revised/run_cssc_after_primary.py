"""Complete dependent CSSC analyses after the revised primary decisions pass."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REV = ROOT / "CSSC_REVIEW_REVISION_20261112"
CODE = ROOT / "exp1" / "code"
REVISED = CODE / "revised"
PRIMARY = REV / "primary_cv_crps_decisions_20261115"
AUDIT_ROOT = PRIMARY / "support_policy_audit" / "b92088b6a7e3c5ae"
SOURCE_MANIFEST = AUDIT_ROOT / "audit_manifest.json"
ANALYSIS = REV / "primary_cv_crps_analysis_20261115" / "analysis"
BOUNDS = REV / "primary_cv_crps_primitive_bounds_20261115"
HIGH = REV / "high_pstar_cv_crps_20261115"
LOGS = REV / "postprocess_logs"
STATE = LOGS / "pipeline_state.json"


def write_state(status: str, stage: str, **kwargs: object) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "stage": stage, "time_local": time.strftime("%Y-%m-%d %H:%M:%S %z"), **kwargs}
    temp = STATE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(STATE)


def execute(stage: str, command: list[str]) -> None:
    log = LOGS / f"{stage}.log"
    write_state("RUNNING", stage, command=command, log=str(log))
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{stage} exited {result.returncode}; see {log}")


def passed(path: Path) -> bool:
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("status") == "PASS"


def analysis_done(path: Path) -> bool:
    if not path.is_file():
        return False
    checks = json.loads(path.read_text(encoding="utf-8"))
    return (checks.get("decision_rows") == 450000
            and checks.get("unique_histories") == 10000
            and checks.get("regret_decomposition_failures") == 0
            and checks.get("all_outputs_created") is True)


def focal_done(path: Path) -> bool:
    if not path.is_file():
        return False
    checks = json.loads(path.read_text(encoding="utf-8"))
    return (checks.get("source_rows") == 450000
            and checks.get("bootstrap_resamples") == 5000
            and (path.parent / "focal_maxt_intervals.csv").is_file())


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    write_state("WAITING", "primary_decisions", source_manifest=str(SOURCE_MANIFEST))
    while True:
        if SOURCE_MANIFEST.is_file():
            source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
            if source.get("status") == "FAIL":
                raise RuntimeError("Revised primary decision run failed")
            if source.get("status") == "PASS":
                if len(source.get("chunks", [])) != 400:
                    raise RuntimeError("Primary manifest says PASS without all 400 chunks")
                break
        time.sleep(30)
    merged = Path(source["merged_decisions"]).resolve()
    if not merged.is_file():
        raise FileNotFoundError(merged)
    decision_audit = PRIMARY / "full_audit.json"
    if not passed(decision_audit):
        execute("primary_full_audit", [sys.executable, str(REVISED / "audit_primary_cv_crps_decisions.py"),
                "--decision-manifest", str(SOURCE_MANIFEST),
                "--selector-run", str(REV / "primary_cv_crps_20261115"),
                "--output", str(decision_audit), "--require-complete"])
    analysis_checks = ANALYSIS / "primary_analysis_checks.json"
    if not analysis_done(analysis_checks):
        execute("primary_analysis", [sys.executable, str(REVISED / "analyze_primary_no_nominal_ks.py"),
                "--base-repo", str(CODE / "legacy_engine"), "--data", str(PRIMARY),
                "--out", str(ANALYSIS), "--expected-replications", "10000"])
    focal = ANALYSIS / "audit" / "audit_summary.json"
    if not focal_done(focal):
        execute("primary_focal_audit", [sys.executable, str(CODE / "audit_cssc_primary_review.py"),
                "--source", str(ANALYSIS / "primary_decisions_enriched_10000.csv.gz"),
                "--output", str(ANALYSIS / "audit")])
    if not passed(HIGH / "high_pstar_smoke_validation.json"):
        execute("high_pstar_smoke", [sys.executable, str(REVISED / "run_high_pstar_cv_crps.py"),
                "--workers", "1", "--smoke"])
    high_command = [sys.executable, str(REVISED / "run_high_pstar_cv_crps.py"),
                    "--workers", "5", "--resume"]
    bound_command = [sys.executable, str(REVISED / "compute_primitive_bounds.py"),
                     "--base-repo", str(CODE / "legacy_engine"),
                     "--audit-script", str(CODE / "legacy" / "run_support_policy_audit.py"),
                     "--audit-root", str(AUDIT_ROOT), "--decision-file", str(merged),
                     "--out", str(BOUNDS), "--workers", "5"]
    if (BOUNDS / "primitive_bounds_manifest.json").is_file():
        bound_command.append("--resume")
    active = []
    for stage, command, expected in (
        ("high_pstar_full", high_command, HIGH / "high_pstar_run_manifest.json"),
        ("primitive_bounds_full", bound_command, BOUNDS / "primitive_bounds_manifest.json"),
    ):
        if passed(expected):
            continue
        log = LOGS / f"{stage}.log"
        output = log.open("w", encoding="utf-8")
        proc = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        active.append((stage, proc, output, log, expected))
    write_state("RUNNING", "independent_full_runs", active=[x[0] for x in active])
    failures = []
    for stage, proc, output, log, expected in active:
        code = proc.wait()
        output.close()
        if code or not passed(expected):
            failures.append(f"{stage}: exit={code}; log={log}")
    if failures:
        raise RuntimeError("; ".join(failures))
    write_state("PASS", "dependent_analyses_complete", primary_audit=str(decision_audit),
                analysis=str(analysis_checks), high_pstar=str(HIGH / "high_pstar_run_manifest.json"),
                numerical_bounds=str(BOUNDS / "primitive_bounds_manifest.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_state("FAIL", "pipeline", error=repr(exc), traceback=traceback.format_exc())
        raise
