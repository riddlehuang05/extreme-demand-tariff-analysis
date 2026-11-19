"""Build CSSC portal files with ASCII names and sub-128 MB archives."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "CSSC_SUBMISSION_20261112"
REV = ROOT / "CSSC_REVIEW_REVISION_20261112"
DATA = REV / "source_data"
PORTAL = ROOT / "CSSC_PORTAL_20261115"
MAX_BYTES = 120_000_000
HISTORY_SOURCE = (ROOT.parent / "！初稿V1.0" / "修改V1.2" / "修改代码+数据"
                  / "data" / "primary_histories_offline_20261001")
EXTERNAL_CODE_ROOT = (ROOT.parent / "！初稿V1.0" / "修改V1.2" / "补充数据"
                      / "recovered_outside_submission" / "external_factory_reanalysis"
                      / "run_ready_project" / "01_Code")
EXTERNAL_OUTPUT_ROOT = (ROOT.parent / "！初稿V1.0" / "修改文件夹"
                        / "交接包_终版V1.1_完整代码数据MD_20260917"
                        / "02_V1.1补实验" / "02_Data" / "outputs" / "external_rolling_cv")
EXTERNAL_CHARGE_CODE = (ROOT.parent / "！初稿V1.0" / "修改V1.2"
                        / "修改代码+数据" / "code"
                        / "complete_external_support_reanalysis.py")
CV_RUN = REV / "primary_cv_crps_20261115"
DECISION_RUN = REV / "primary_cv_crps_decisions_20261115"
PRIMARY_ANALYSIS = REV / "primary_cv_crps_analysis_20261115" / "analysis"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_completed_primary() -> None:
    """Prevent a portal archive that mixes the new method with old results."""
    selector = json.loads((CV_RUN / "full_audit.json").read_text(encoding="utf-8"))
    if (selector.get("status") != "PASS" or selector.get("history_count") != 10000
            or selector.get("candidate_fold_rows") != 160000
            or selector.get("selection_disagreements") != 0):
        raise ValueError("four-fold CV-CRPS selector has not passed its full audit")
    manifests = list((DECISION_RUN / "support_policy_audit").glob("*/audit_manifest.json"))
    if len(manifests) != 1:
        raise ValueError("expected one completed four-fold primary decision run")
    decision = json.loads(manifests[0].read_text(encoding="utf-8"))
    if (decision.get("status") != "PASS" or len(decision.get("chunks", [])) != 400
            or sum(c.get("decision_rows", 0) for c in decision["chunks"]) != 450000):
        raise ValueError("four-fold primary decisions are incomplete")
    decision_audit = json.loads((DECISION_RUN / "full_audit.json").read_text(encoding="utf-8"))
    if (decision_audit.get("status") != "PASS"
            or decision_audit.get("verified_histories") != 10000
            or decision_audit.get("verified_decision_rows") != 450000
            or decision_audit.get("verified_fit_records") != 50000):
        raise ValueError("four-fold primary decisions have not passed the full audit")
    merged = Path(decision.get("merged_decisions", ""))
    if not merged.is_file() or sha(merged) != decision.get("merged_decisions_sha256"):
        raise ValueError("four-fold merged decisions are missing or have a hash mismatch")
    checks = json.loads((PRIMARY_ANALYSIS / "primary_analysis_checks.json").read_text(encoding="utf-8"))
    if (checks.get("decision_rows") != 450000 or checks.get("unique_histories") != 10000
            or checks.get("regret_decomposition_failures") != 0
            or not checks.get("all_outputs_created")):
        raise ValueError("revised primary analysis has not passed its integrity checks")
    with (PRIMARY_ANALYSIS / "primary_functional_fidelity.csv").open(encoding="utf-8", newline="") as stream:
        fidelity = list(csv.DictReader(stream))
    if (len(fidelity) != 5 or {row["method_id"] for row in fidelity}
            != {"TAIL", "EVENT-EMP", "GEV", "KDE", "EMP"}
            or any(int(row["histories"]) != 10000 for row in fidelity)):
        raise ValueError("revised primary functional-fidelity summary is incomplete")
    enriched = PRIMARY_ANALYSIS / "primary_decisions_enriched_10000.csv.gz"
    publication = DATA / "data" / "source" / enriched.name
    if not enriched.is_file() or not publication.is_file() or sha(enriched) != sha(publication):
        raise ValueError("publication decision table differs from revised primary analysis")
    figure_audit = json.loads((DATA / "audit" / "cssc_primary_figures_layout.json").read_text(encoding="utf-8"))
    if figure_audit.get("status") != "PASS":
        raise ValueError("revised primary figure layout has not passed")
    if sha(DATA / "data" / "source" / "primary" / "primary_analysis_checks.json") != figure_audit.get("source_analysis_sha256"):
        raise ValueError("primary figure source analysis differs from publication data")
    for name, expected in figure_audit.get("pdf_sha256", {}).items():
        if sha(SUB / "Manuscript" / "figures" / name) != expected:
            raise ValueError(f"primary figure PDF changed after validation: {name}")
    if set(figure_audit.get("pdf_sha256", {})) != {"Figure_1.pdf", "Figure_3.pdf"}:
        raise ValueError("primary figure audit does not cover both regenerated figures")
    cross_manifest = REV / "crossed_grid_cv_crps_20261115" / "raw" / "run_manifest.json"
    cross_raw = REV / "crossed_grid_cv_crps_20261115" / "raw" / "crossed_grid_no_upper_decisions.csv.gz"
    if not cross_manifest.is_file() or not cross_raw.is_file():
        raise ValueError("revised crossed-grid run is missing")
    cross = json.loads(cross_manifest.read_text(encoding="utf-8"))
    if (cross.get("status") != "PASS" or cross.get("replications") != 1000
            or cross.get("rows") != 380000 or cross.get("cells") != 76
            or sha(cross_raw) != cross.get("decisions_sha256")):
        raise ValueError("revised crossed-grid run has not passed its hash audit")
    figure2_audit = DATA / "audit" / "figure2_primary_crossed_qc.json"
    if not figure2_audit.is_file():
        raise ValueError("revised Figure 2 audit is missing")
    figure2 = json.loads(figure2_audit.read_text(encoding="utf-8"))
    if (figure2.get("raw_crossed_decision_sha256") != cross["decisions_sha256"]
            or figure2.get("crossed_run_manifest_sha256") != sha(cross_manifest)
            or figure2.get("figure2_pdf_sha256") != sha(SUB / "Manuscript/figures/Figure_2.pdf")
            or figure2.get("figure_s8", {}).get("figure_s8_pdf_sha256")
            != sha(SUB / "Supporting_Information/figures/FigureS8.pdf")):
        raise ValueError("revised crossed-grid figures differ from audited data")
    nb_root = REV / "nb_weibull_cv_crps_20261115"
    nb_manifest_path = nb_root / "run_manifest.json"
    nb = json.loads(nb_manifest_path.read_text(encoding="utf-8"))
    if (nb.get("status") != "PASS" or nb.get("replications_requested") != 10000
            or nb.get("summary_validation", {}).get("status") != "PASS"
            or nb.get("summary_validation", {}).get("decision_rows") != 450000):
        raise ValueError("revised NB--Weibull stress experiment is incomplete")
    nb_table_audit = json.loads((nb_root / "tables_refresh_audit.json").read_text(encoding="utf-8"))
    if (nb_table_audit.get("status") != "PASS"
            or nb_table_audit.get("run_manifest_sha256") != sha(nb_manifest_path)):
        raise ValueError("revised NB--Weibull tables are not hash-aligned")
    for name, expected in nb_table_audit.get("table_sha256", {}).items():
        if sha(SUB / "Supporting_Information" / "tables" / name) != expected:
            raise ValueError(f"revised NB--Weibull table changed after audit: {name}")
    nb_figure = json.loads((nb_root / "figure_s7_audit.json").read_text(encoding="utf-8"))
    if (nb_figure.get("status") != "PASS" or nb_figure.get("rows") != 450000
            or nb_figure.get("run_manifest_sha256") != sha(nb_manifest_path)
            or nb_figure.get("figure_pdf_sha256")
            != sha(SUB / "Supporting_Information/figures/FigureS7.pdf")):
        raise ValueError("revised Figure S7 differs from NB--Weibull run")
    bounds_root = REV / "primary_cv_crps_primitive_bounds_20261115"
    bounds = json.loads((bounds_root / "primitive_bounds_manifest.json").read_text(encoding="utf-8"))
    bounds_summary = json.loads((bounds_root / "primitive_bounds_summary.json").read_text(encoding="utf-8"))
    if (bounds.get("status") != "PASS" or len(bounds.get("tasks", [])) != 400
            or bounds_summary.get("status") != "PASS"
            or bounds_summary.get("n_decision_rows") != 450000):
        raise ValueError("revised primitive and Wasserstein numerical audit is incomplete")
    validation_dir = DATA / "validation"
    if ({p.name for p in validation_dir.iterdir() if p.is_file()}
            != {"primary_analysis_checks.json", "primitive_bounds_summary.json"}
            or sha(validation_dir / "primary_analysis_checks.json")
            != sha(PRIMARY_ANALYSIS / "primary_analysis_checks.json")
            or sha(validation_dir / "primitive_bounds_summary.json")
            != sha(bounds_root / "primitive_bounds_summary.json")):
        raise ValueError("published validation files are stale or hash-mismatched")
    w1_audit_dir = DATA / "data/derived_publication/tariff_contrast_wasserstein"
    w1_audit = json.loads((w1_audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    if (w1_audit.get("status") != "PASS" or w1_audit.get("decision_rows") != 450000
            or w1_audit.get("source_manifest_sha256") != sha(bounds_root / "primitive_bounds_manifest.json")
            or w1_audit.get("figure_pdf_sha256") != sha(SUB / "Supporting_Information/figures/FigureS3.pdf")):
        raise ValueError("revised Figure S3 differs from numerical-bound source")
    high_root = REV / "high_pstar_cv_crps_20261115"
    high_manifest_path = high_root / "high_pstar_run_manifest.json"
    high_manifest = json.loads(high_manifest_path.read_text(encoding="utf-8"))
    high_check = json.loads((high_root / "high_pstar_checks.json").read_text(encoding="utf-8"))
    high_audit = json.loads((high_root / "figure_s1_table_s1_validation.json").read_text(encoding="utf-8"))
    high_published = DATA / "data/source/high_pstar"
    if (high_manifest.get("status") != "PASS" or high_manifest.get("chunks_completed") != 400
            or len(high_manifest.get("chunk_records", [])) != 400
            or high_check.get("status") != "PASS"
            or high_check.get("decision_rows_reanalyzed") != 1_500_000
            or high_audit.get("status") != "PASS" or high_audit.get("histories") != 10000
            or high_audit.get("decisions") != 1_500_000
            or high_audit.get("source_manifest_sha256") != sha(high_manifest_path)
            or sha(high_published / "high_pstar_run_manifest.json") != sha(high_manifest_path)):
        raise ValueError("revised high-p* stress or publication archive is incomplete")
    expected_high_outputs = {
        str(high_root / "high_pstar_same_information_paired_contrasts.csv"):
            high_root / "high_pstar_same_information_paired_contrasts.csv",
        str(SUB / "Supporting_Information/tables/Table_S1.tex"):
            SUB / "Supporting_Information/tables/Table_S1.tex",
        str(SUB / "Supporting_Information/figures/FigureS1.pdf"):
            SUB / "Supporting_Information/figures/FigureS1.pdf",
    }
    for name, path in expected_high_outputs.items():
        if high_audit.get("outputs", {}).get(name) != sha(path):
            raise ValueError(f"revised high-p* output changed after audit: {path.name}")
    for record in high_manifest["chunk_records"]:
        start, end = record["start"], record["end"]
        name = f"chunk_{start:05d}_{end:05d}_regrets.npz"
        source = high_root / "checkpoints" / name
        published = high_published / "regret_checkpoints" / name
        if sha(source) != record["regrets_sha256"] or sha(published) != record["regrets_sha256"]:
            raise ValueError(f"revised high-p* regret checkpoint is missing or changed: {name}")


def zip_files(path: Path, entries: list[tuple[Path, str]]) -> None:
    if path.exists():
        path.unlink()
    with zipfile.ZipFile(path, "w", allowZip64=True, compression=zipfile.ZIP_STORED) as archive:
        for source, name in entries:
            archive.write(source, name)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"Archive exceeds upload limit: {path.name} ({path.stat().st_size})")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"ZIP integrity check failed: {path.name}")


def tree_entries(base: Path, files: list[Path], prefix: str = "") -> list[tuple[Path, str]]:
    return [(file, prefix + file.relative_to(base).as_posix()) for file in sorted(files)]


def build_history_parts() -> None:
    event = HISTORY_SOURCE / "raw" / "event_clusters_10000.parquet"
    monthly = HISTORY_SOURCE / "raw" / "monthly_maxima_10000.parquet"
    manifest = HISTORY_SOURCE / "offline_history_manifest.json"
    if not all(path.is_file() for path in (event, monthly, manifest)):
        raise FileNotFoundError("complete primary history release is unavailable")
    history = json.loads(manifest.read_text(encoding="utf-8"))
    for key, path in (("event_clusters", event), ("monthly_maxima", monthly)):
        if sha(path) != history["outputs"][key + "_sha256"]:
            raise ValueError(f"source history hash mismatch: {key}")
    chunk_bytes = 95_000_000
    if not 4 * chunk_bytes < event.stat().st_size <= 5 * chunk_bytes:
        raise ValueError("expected event history to fit exactly five parts")
    consumed = 0
    with event.open("rb") as stream:
        for i in range(1, 6):
            path = PORTAL / f"CSSC_History_Part{i}.zip"
            with zipfile.ZipFile(path, "w", allowZip64=True, compression=zipfile.ZIP_STORED) as archive:
                with archive.open(f"history_parts/event_clusters_10000.parquet.part{i:02d}", "w") as part:
                    remaining = chunk_bytes
                    while remaining:
                        block = stream.read(min(1 << 20, remaining))
                        if not block:
                            break
                        part.write(block)
                        remaining -= len(block)
                        consumed += len(block)
                if i == 1:
                    archive.write(monthly, "raw/monthly_maxima_10000.parquet")
                    archive.write(manifest, "offline_history_manifest.json")
            if path.stat().st_size > MAX_BYTES:
                raise ValueError(f"history archive exceeds upload limit: {path}")
    if consumed != event.stat().st_size:
        raise AssertionError("history archive did not consume the complete event file")


def refresh_source_index() -> None:
    old_path = DATA / "SOURCE_DATA_INDEX.csv"
    old = {}
    if old_path.exists():
        with old_path.open(encoding="utf-8-sig", newline="") as stream:
            old = {row["relative_path"]: row for row in csv.DictReader(stream)}
    files = [p for p in DATA.rglob("*")
             if p.is_file() and p.name != old_path.name and "__pycache__" not in p.parts
             and not {"high_density_v2", "r5_primitive_bounds", "tables", "code"}.intersection(p.relative_to(DATA).parts)]
    rows = []
    for file in sorted(files, key=lambda p: p.relative_to(DATA).as_posix()):
        rel = file.relative_to(DATA).as_posix()
        prior = old.get(rel, {})
        if rel.startswith("data/source/high_pstar/"):
            owner, usage = "revised high-p* tariff stress", "saved-law re-evaluation and paired history-level regret"
        elif rel.startswith("data/source/nonpoisson_gpd_challenge/"):
            owner, usage = "revised NB-Weibull challenge", "independent 10,000-history stress analysis"
        elif rel.startswith("data/source/crossed_grid_no_upper_1000/"):
            owner, usage = "revised crossed tariff grid", "1,000-history 76-cell decision analysis"
        elif rel.startswith("data/numerical_audit/") or rel.startswith("data/derived_publication/tariff_contrast_wasserstein/"):
            owner, usage = "revised numerical-bound audit", "50,000 fitted laws and 450,000 decision-row checks"
        elif rel.startswith("data/source/primary/") or rel == "data/source/primary_decisions_enriched_10000.csv.gz":
            owner, usage = "revised primary experiment", "10,000-history paired decision analysis"
        else:
            owner = prior.get("scientific_owner", "archive documentation or executable source")
            usage = prior.get("usage", "supports manuscript reproduction and integrity checks")
        rows.append({
            "relative_path": rel,
            "size": str(file.stat().st_size),
            "scientific_owner": owner,
            "usage": usage,
        })
    with old_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "size", "scientific_owner", "usage"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    require_completed_primary()
    PORTAL.mkdir(exist_ok=True)
    selector_files = [CV_RUN / "run_manifest.json", CV_RUN / "full_audit.json"]
    for path in sorted(CV_RUN.glob("cv_*.csv")):
        parts = path.stem.split("_")
        if len(parts) == 4 and int(parts[2]) - int(parts[1]) == 99:
            selector_files.append(path)
    if len(selector_files) != 202:
        raise ValueError(f"expected 202 four-fold selection source files, found {len(selector_files)}")
    refresh_source_index()

    main_dir = REV / "portal_stage" / "MainSource"
    main_dir.mkdir(parents=True, exist_ok=True)
    main_src = SUB / "Manuscript"
    for name in ("Manuscript.tex", "references.bib", "interact.cls", "tfapacite.bst", "tfapastyle.sty", "tfs.bst"):
        source = main_src / name
        if source.exists():
            shutil.copy2(source, main_dir / name)
    shutil.copytree(main_src / "figures", main_dir / "figures", dirs_exist_ok=True)
    zip_files(PORTAL / "CSSC_Manuscript_Source.zip", tree_entries(main_dir, [p for p in main_dir.rglob("*") if p.is_file()]))

    si_dir = REV / "portal_stage" / "SupplementarySource"
    si_src = SUB / "Supporting_Information"
    if si_dir.exists():
        if not si_dir.resolve().is_relative_to((REV / "portal_stage").resolve()):
            raise ValueError("Supplementary stage is outside the intended workspace")
        shutil.rmtree(si_dir)
    si_dir.mkdir(parents=True)
    for name in ("Supporting_Information.tex", "FIGURES.tex", "interact.cls", "README.txt"):
        source = si_src / name
        if source.exists():
            shutil.copy2(source, si_dir / name)
    for folder in ("figures", "tables"):
        shutil.copytree(si_src / folder, si_dir / folder, dirs_exist_ok=True)
    zip_files(PORTAL / "CSSC_Supplementary_Source.zip", tree_entries(si_dir, [p for p in si_dir.rglob("*") if p.is_file()]))

    shutil.copy2(SUB / "Manuscript" / "Manuscript.pdf", PORTAL / "CSSC_Main_Manuscript.pdf")
    shutil.copy2(SUB / "Supporting_Information" / "Supporting_Information.pdf", PORTAL / "CSSC_Supplementary_Information.pdf")
    shutil.copy2(REV / "CSSC_Cover_Letter.txt", PORTAL / "CSSC_Cover_Letter.txt")

    for i in range(1, 5):
        pdf = SUB / "Manuscript" / "figures" / f"Figure_{i}.pdf"
        eps = PORTAL / f"CSSC_Figure_{i}.eps"
        subprocess.run(["pdftops", "-eps", str(pdf), str(eps)], check=True, capture_output=True)
    for i in range(1, 9):
        pdf = SUB / "Supporting_Information" / "figures" / f"FigureS{i}.pdf"
        eps = PORTAL / f"CSSC_Figure_S{i}.eps"
        subprocess.run(["pdftops", "-eps", str(pdf), str(eps)], check=True, capture_output=True)

    code_root = ROOT / "exp1" / "code"
    code_files = [p for p in (code_root / "legacy_engine").rglob("*")
                  if p.is_file() and p.suffix.lower() in {".py", ".yaml"}]
    code_files += [p for p in (code_root / "legacy").glob("run_support_policy_audit.py")]
    code_files += [p for p in (code_root / "revised").glob("*.py")]
    code_files += [code_root / name for name in (
        "audit_cssc_primary_review.py", "audit_cssc_selector_sensitivity.py",
        "build_cssc_portal_package.py",
        "compose_cssc_submission.py", "apply_cssc_review_round.py",
        "compress_cssc_main.py", "polish_cssc_narrative.py")]
    code_files += [ROOT / "exp1" / "config" / name for name in (
        "exp1_protocol.json", "exp1_execution_addendum_20261107.json",
        "cssc_primary_cv_crps_amendment_20261115.json")]
    code_files += [ROOT / "exp1" / "requirements_cssc.txt",
                   ROOT / "exp1" / "CSSC_Reproducibility_README.md"]
    code_files += [ROOT / "exp1" / "outputs" / "raw" / "history_registry_pilot100_parallel.csv",
                   ROOT / "exp1" / "outputs" / "raw" / "decision_rows_pilot100_parallel.csv.gz",
                   REV / "audit_event_emp_ties.py", REV / "event_emp_tie_audit.json"]
    code_files += [p for p in (DATA / "validation").rglob("*") if p.is_file()]
    code_files += [p for p in (PRIMARY_ANALYSIS / "audit").glob("*")
                   if p.is_file() and p.suffix.lower() in {".json", ".csv"}]
    code_files += [EXTERNAL_CODE_ROOT / "scripts" / "run_external_rolling_validation.py",
                   EXTERNAL_CODE_ROOT / "configs" / "external_rolling_validation.yaml"]
    code_files.append(EXTERNAL_CHARGE_CODE)
    code_files += [p for p in (EXTERNAL_CODE_ROOT / "vendor_track_a").rglob("*.py")]
    code_files += [p for p in (EXTERNAL_CODE_ROOT / "inputs" / "korea_gate6_data_full").glob("*")
                   if p.is_file()]
    code_files += [EXTERNAL_OUTPUT_ROOT / name for name in (
        "prediction_scores.csv", "fit_diagnostics.csv", "candidate_rolling_cv.csv",
        "factory_cluster_contrasts.csv", "summary.json")]
    code_files += [DATA / "README.md", DATA / "SOURCE_DATA_INDEX.csv", DATA / "SCIENTIFIC_OWNERSHIP.json", DATA / "DERIVATION_MAP_TARIFF_CONTRAST_W1.md"]
    code_files += [p for p in (DATA / "audit").rglob("*") if p.is_file()]
    code_entries = []
    for p in sorted(set(code_files)):
        if p.exists() and "__pycache__" not in p.parts:
            if p.is_relative_to(ROOT / "exp1"):
                dest = "exp1/" + p.relative_to(ROOT / "exp1").as_posix()
            elif p.is_relative_to(DATA):
                dest = "source_data/" + p.relative_to(DATA).as_posix()
            elif p.is_relative_to(EXTERNAL_CODE_ROOT):
                dest = "external/" + p.relative_to(EXTERNAL_CODE_ROOT).as_posix()
            elif p.is_relative_to(EXTERNAL_OUTPUT_ROOT):
                dest = "external/outputs/external_rolling_cv/" + p.name
            elif p == EXTERNAL_CHARGE_CODE:
                dest = "external/scripts/complete_external_support_reanalysis.py"
            else:
                dest = "audit/" + p.name
            code_entries.append((p, dest))
    zip_files(PORTAL / "CSSC_Reproducibility_Code.zip", code_entries)

    audit_files = sorted((DATA / "data" / "numerical_audit").glob("*.parquet"))
    audit_bytes = 0
    audit_mid = 0
    for path in audit_files:
        if audit_bytes + path.stat().st_size > 40_000_000:
            break
        audit_bytes += path.stat().st_size
        audit_mid += 1
    part_a = [p for p in (DATA / "data" / "source" / "high_pstar").rglob("*") if p.is_file()]
    zip_files(PORTAL / "CSSC_Data_Part1.zip", tree_entries(DATA, part_a, "source_data/"))

    part_b = [DATA / "data" / "source" / "primary_decisions_enriched_10000.csv.gz"] + audit_files[:audit_mid]
    zip_files(PORTAL / "CSSC_Data_Part2.zip", tree_entries(DATA, part_b, "source_data/"))

    part_c = audit_files[audit_mid:]
    for folder in ("crossed_grid_no_upper_1000", "dgp_external_support", "primary", "s6_alpha120", "nonpoisson_gpd_challenge", "teacher_sonnet"):
        part_c += [p for p in (DATA / "data" / "source" / folder).rglob("*") if p.is_file()]
    part_c += [p for p in (DATA / "data" / "plot").rglob("*")
               if p.is_file() and "high_density_v2" not in p.parts and "r5_primitive_bounds" not in p.parts]
    part_c += [p for p in (DATA / "data" / "derived_publication").rglob("*") if p.is_file()]
    part_c += [p for p in (DATA / "data" / "reported_separate_experiments").rglob("*") if p.is_file()]
    covered = set(part_a + part_b + part_c)
    stale_dirs = {"high_density_v2", "r5_primitive_bounds"}
    part_c += [p for p in (DATA / "data").rglob("*")
               if p.is_file() and p not in covered and not stale_dirs.intersection(p.parts)]
    zip_files(PORTAL / "CSSC_Data_Part3.zip", tree_entries(DATA, part_c, "source_data/"))
    zip_files(PORTAL / "CSSC_Data_Part4.zip",
              tree_entries(CV_RUN, selector_files, "primary_cv_crps/"))
    decision_manifest = next((DECISION_RUN / "support_policy_audit").glob("*/audit_manifest.json"))
    zip_files(PORTAL / "CSSC_Data_Part5.zip", [
        (decision_manifest, "primary_decisions/audit_manifest.json"),
        (DECISION_RUN / "full_audit.json", "primary_decisions/full_audit.json"),
    ])
    analysis_files = [p for p in PRIMARY_ANALYSIS.iterdir()
                      if p.is_file() and p.suffix.lower() in {".json", ".csv"}]
    analysis_files += [p for p in (PRIMARY_ANALYSIS / "audit").iterdir()
                       if p.is_file() and p.suffix.lower() in {".json", ".csv"}]
    associations = REV / "primary_cv_crps_analysis_20261115" / "associations"
    analysis_files += [p for p in associations.iterdir()
                       if p.is_file() and p.suffix.lower() in {".json", ".csv"}]
    if not analysis_files:
        raise ValueError("revised primary analysis outputs are missing")
    analysis_entries = tree_entries(PRIMARY_ANALYSIS, [p for p in analysis_files
                                                      if p.is_relative_to(PRIMARY_ANALYSIS)],
                                    "primary_analysis/")
    analysis_entries += tree_entries(associations, [p for p in analysis_files
                                                   if p.is_relative_to(associations)],
                                     "primary_analysis/associations/")
    zip_files(PORTAL / "CSSC_Data_Part6.zip", analysis_entries)
    build_history_parts()

    readme = """CSSC submission files

Upload CSSC_Main_Manuscript.pdf as the main manuscript and
CSSC_Supplementary_Information.pdf as supplementary information.
CSSC_Submission_Core.zip is a local handoff copy of the non-data files;
upload the individual files from this directory, not this duplicate bundle.
Upload CSSC_Figure_1.eps through CSSC_Figure_4.eps as separate figures;
upload CSSC_Figure_S1.eps through CSSC_Figure_S8.eps if the portal requests
separate supplementary figures. The two source ZIPs and reproducibility
archives are supporting files. Each archive is below the portal's 128 MB
single-file limit. Data Parts 1-3 preserve paths under source_data/;
extract them into the same directory to reconstruct the publication archive.
Data Part 1 contains revised high-p* summaries and paired regret checkpoints.
Data Part 4 contains all 10,000 four-fold threshold choices, 160,000
candidate-fold scores, and the full-run audit. Data Part 5 contains the
primary decision-run manifest and audit. Data Part 6 contains revised
primary analysis outputs. Data Part 2 contains the revised 450,000-row
decision table.
The five history ZIPs contain the complete 10,000-history primary inputs.
Run revised/reconstruct_cssc_histories.py from the code ZIP with --portal
pointing to this directory and --output pointing to a new history directory;
the script verifies both reconstructed Parquet files against their manifest.
From the extracted code ZIP, the full corrected primary run can be started as:
python exp1/code/revised/run_primary_no_nominal_ks.py --base-repo
exp1/code/legacy_engine --data HISTORY_DIR --run-manifest
HISTORY_DIR/offline_history_manifest.json --output PRIMARY_DIR --start 0
--end 10000 --chunk-size 25 --workers 8
Replace HISTORY_DIR and PRIMARY_DIR with the chosen directories. The runner
checks source hashes, selects thresholds by four-fold held-out CRPS, runs
500-resample shape diagnostics, and exports a merged primary
decision table under support_policy_audit/.
The reproducibility-code ZIP also contains the external rolling-validation
runner, configuration, derived Korean inputs, 3,456 candidate-fold records,
224 prediction records, fit diagnostics, and their run-summary checksums.

The cover letter is included as a concise submission aid. The CSSC
instructions require an originality and no-simultaneous-submission
statement; the corresponding author must verify that declaration.

The revised source-data index records current file sizes. SHA-256 values for
every portal file are listed in CSSC_Portal_Manifest.csv.
"""
    (PORTAL / "CSSC_Upload_Readme.txt").write_text(readme, encoding="utf-8")
    core = PORTAL / "CSSC_Submission_Core.zip"
    core_files = [p for p in PORTAL.iterdir() if p.is_file() and not p.name.startswith(("CSSC_Data_Part", "CSSC_History_Part"))
                  and p.name not in {core.name, "CSSC_Portal_Manifest.csv"}]
    zip_files(core, [(p, p.name) for p in sorted(core_files)])
    artifacts = sorted(p for p in PORTAL.iterdir() if p.is_file() and p.name != "CSSC_Portal_Manifest.csv")
    with (PORTAL / "CSSC_Portal_Manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filename", "bytes", "sha256"])
        for path in artifacts:
            writer.writerow([path.name, path.stat().st_size, sha(path)])
    print(json.dumps({"portal": str(PORTAL), "files": len(artifacts) + 1,
                      "max_bytes": max(p.stat().st_size for p in PORTAL.iterdir() if p.is_file()),
                      "archives": {p.name: p.stat().st_size for p in PORTAL.glob("*.zip")}}, indent=2))


if __name__ == "__main__":
    main()
