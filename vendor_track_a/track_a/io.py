"""Deterministic structured-output helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sha256_files(paths: Iterable[str | Path], root: str | Path) -> str:
    root_path = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted((Path(item).resolve() for item in paths), key=lambda item: item.as_posix()):
        relative = path.relative_to(root_path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_matching_smoke(smoke_summary: str | Path, run_signature: str) -> dict[str, Any]:
    path = Path(smoke_summary)
    if not path.exists():
        raise RuntimeError(f"full run refused: smoke evidence is missing at {path}")
    record = read_json(path)
    if record.get("status") != "PASS":
        raise RuntimeError("full run refused: latest smoke status is not PASS")
    if record.get("run_signature") != run_signature:
        raise RuntimeError("full run refused: code/config changed since smoke; rerun smoke")
    return record


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


class AtomicCsvWriter:
    """Incremental CSV writer that publishes only a completely closed file."""

    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.destination = Path(path)
        self.temporary = self.destination.with_suffix(self.destination.suffix + ".tmp")
        self.fieldnames = fieldnames
        self._handle: Any = None
        self._writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> "AtomicCsvWriter":
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temporary.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=self.fieldnames, extrasaction="raise"
        )
        self._writer.writeheader()
        return self

    def writerow(self, row: Mapping[str, Any]) -> None:
        if self._writer is None:
            raise RuntimeError("CSV writer is not open")
        self._writer.writerow(row)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.close()
        if exc_type is None:
            os.replace(self.temporary, self.destination)
        elif self.temporary.exists():
            self.temporary.unlink()


def environment_record() -> dict[str, str]:
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
