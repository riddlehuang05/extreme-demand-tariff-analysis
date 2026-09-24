"""Reconstruct the external rolling-validation input from the cited source."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extreme_demand.external.korea import (  # noqa: E402
    load_dr_intervals,
    monthly_eligibility,
    reconstruct_factory,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Reconstruct one factory")
    parser.add_argument("--source-root", type=Path, help="Root containing Factories and DR_information")
    parser.add_argument("--output-root", type=Path, help="Directory for reconstructed inputs")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs" / "external_korea.yaml").read_text(encoding="utf-8"))["external_korea"]
    source = (args.source_root or ROOT / config["source_root_relative"]).resolve()
    if not source.is_dir():
        parser.error(f"source directory does not exist: {source}")
    config["source_root"] = source
    intervals = load_dr_intervals(config)
    factories = config["smoke_factories"] if args.smoke else config["factories"]
    output = (args.output_root or ROOT / "inputs" / "korea_gate6_data_full").resolve()
    output.mkdir(parents=True, exist_ok=True)

    fixed_parts = []
    eligibility_parts = []
    manifest = []
    workbook = source / config["dr_workbook"]
    manifest.append({"source_type": "dr_workbook", "factory": "", "path_relative_to_source_root": workbook.relative_to(source).as_posix(), "bytes": workbook.stat().st_size, "sha256": sha256(workbook)})
    for factory in factories:
        path = source / config["factory_subdirectory"] / f"{factory}.csv"
        manifest.append({"source_type": "factory_minute_csv", "factory": factory, "path_relative_to_source_root": path.relative_to(source).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
        fixed, sliding, _, _ = reconstruct_factory(path, factory, config, intervals)
        fixed_parts.append(fixed)
        eligibility_parts.append(pd.concat([fixed, sliding], ignore_index=True))

    fixed = pd.concat(fixed_parts, ignore_index=True)
    eligibility = monthly_eligibility(pd.concat(eligibility_parts, ignore_index=True), config)
    fixed.to_parquet(output / "fixed15.parquet", index=False, compression="zstd")
    eligibility.to_csv(output / "monthly_eligibility.csv", index=False)
    pd.DataFrame(manifest).to_csv(output / "input_manifest.csv", index=False)
    print(f"Reconstructed {len(factories)} factories and {len(fixed)} fixed-window rows in {output}")


if __name__ == "__main__":
    main()
