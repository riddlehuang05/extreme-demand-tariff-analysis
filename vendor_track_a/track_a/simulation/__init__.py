"""Streaming simulation and truth-library construction."""

from .truth_library import build_scenario_registry, load_truth_spec, simulate_truth_chunk

__all__ = ["build_scenario_registry", "load_truth_spec", "simulate_truth_chunk"]
