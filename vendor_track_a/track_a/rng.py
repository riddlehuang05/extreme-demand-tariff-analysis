"""Named NumPy SeedSequence child streams with serializable ancestry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


STREAM_NAMES = (
    "calendar",
    "stages",
    "energy",
    "background",
    "rare_events",
    "training_draws",
    "method_mc",
    "bootstrap",
    "test_pools",
)


@dataclass(frozen=True)
class SeedStreamBundle:
    generators: dict[str, np.random.Generator]
    registry_rows: list[dict[str, str | int]]


def make_seed_streams(
    seed_root: int, namespace: Iterable[int], context_id: str
) -> SeedStreamBundle:
    namespace_tuple = tuple(int(value) for value in namespace)
    if seed_root < 0 or any(value < 0 for value in namespace_tuple):
        raise ValueError("seed_root and namespace values must be non-negative")
    parent = np.random.SeedSequence([seed_root, *namespace_tuple])
    children = parent.spawn(len(STREAM_NAMES))
    generators = {
        name: np.random.default_rng(child) for name, child in zip(STREAM_NAMES, children, strict=True)
    }
    rows = [
        {
            "context_id": context_id,
            "stream_name": name,
            "seed_root": seed_root,
            "namespace": ".".join(str(value) for value in namespace_tuple) or "root",
            "parent_entropy": ".".join(str(value) for value in np.atleast_1d(parent.entropy)),
            "spawn_key": ".".join(str(value) for value in child.spawn_key),
        }
        for name, child in zip(STREAM_NAMES, children, strict=True)
    ]
    return SeedStreamBundle(generators=generators, registry_rows=rows)
