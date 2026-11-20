"""Validation helpers for pending-safe LaTeX manuscript builds."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def find_pending_markers(paths: list[Path]) -> list[dict[str, Any]]:
    markers = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "PENDING" not in line:
                continue
            if "\\newcommand{\\resultpending}" in line:
                continue
            markers.append(
                {
                    "path": path.as_posix(),
                    "line": line_number,
                    "text": line.strip(),
                }
            )
    return markers


def latex_log_diagnostics(text: str) -> dict[str, Any]:
    lowered = text.lower()
    return {
        "undefined_reference_warning": "there were undefined references" in lowered,
        "undefined_citation_warning": "citation" in lowered and "undefined" in lowered,
        "overfull_hbox_count": lowered.count("overfull \\hbox"),
        "overfull_vbox_count": lowered.count("overfull \\vbox"),
    }
