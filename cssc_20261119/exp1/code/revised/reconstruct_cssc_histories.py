"""Reassemble the hash-verified primary history files from portal ZIP parts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archives = sorted(args.portal.glob("CSSC_History_Part*.zip"))
    if len(archives) != 5:
        raise ValueError(f"expected five history archives, found {len(archives)}")
    raw = args.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    event = raw / "event_clusters_10000.parquet"
    with event.open("wb") as target:
        for i, archive_path in enumerate(archives, 1):
            name = f"history_parts/event_clusters_10000.parquet.part{i:02d}"
            with zipfile.ZipFile(archive_path) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"corrupt ZIP: {archive_path}")
                with archive.open(name) as part:
                    for block in iter(lambda: part.read(1 << 20), b""):
                        target.write(block)
                if i == 1:
                    for member in ("raw/monthly_maxima_10000.parquet", "offline_history_manifest.json"):
                        destination = args.output / member
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, destination.open("wb") as output:
                            for block in iter(lambda: source.read(1 << 20), b""):
                                output.write(block)
    manifest = json.loads((args.output / "offline_history_manifest.json").read_text(encoding="utf-8"))
    for key, path in (("event_clusters", event), ("monthly_maxima", raw / "monthly_maxima_10000.parquet")):
        expected = manifest["outputs"][key + "_sha256"]
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{key} SHA-256 mismatch: {actual} != {expected}")
    print(json.dumps({"status": "PASS", "event_rows": manifest["event_cluster_rows"],
                      "monthly_rows": manifest["monthly_maxima_rows"],
                      "history_count": manifest["replications"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
