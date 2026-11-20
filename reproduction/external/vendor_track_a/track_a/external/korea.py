"""Leakage-free reconstruction of the ten Korean factory minute series."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


FLAG_COLUMNS = [
    "factory",
    "source_row",
    "timestamp_raw",
    "load_raw",
    "flag_type",
    "action_main",
]


def load_external_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["external_korea"]
    source_root = (Path(project_root) / config["source_root_relative"]).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Korean source root does not exist: {source_root}")
    config["source_root"] = source_root
    return config


def load_dr_intervals(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["source_root"]) / config["dr_workbook"]
    raw = pd.read_excel(path, engine="openpyxl")
    required = {"Industry", "DR Capacity", "Day", "Start time", "End time"}
    if set(raw.columns) != required:
        raise ValueError(f"unexpected DR workbook columns: {list(raw.columns)}")
    name_map = config["factory_name_map"]
    rows = []
    for source_row, row in raw.iterrows():
        factory = name_map.get(str(row["Industry"]))
        if factory is None:
            raise ValueError(f"unmapped DR factory name: {row['Industry']}")
        day = pd.Timestamp(row["Day"]).normalize()
        start_clock = pd.to_timedelta(str(row["Start time"]))
        end_clock = pd.to_timedelta(str(row["End time"]))
        start = day + start_clock
        end = day + end_clock
        if end <= start:
            raise ValueError("DR end must be after start")
        rows.append(
            {
                "source_row": int(source_row) + 2,
                "source_industry": str(row["Industry"]),
                "factory": factory,
                "dr_capacity_source_units": float(row["DR Capacity"]),
                "event_start": start,
                "event_end": end,
                "exclusion_end": end + pd.Timedelta(minutes=int(config["dr_rebound_buffer_minutes"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["factory", "event_start"]).reset_index(drop=True)


def _full_index(config: dict[str, Any]) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(config["start"]), pd.Timestamp(config["end"]), freq="min")


def load_factory_minutes(
    path: str | Path,
    factory: str,
    config: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any], pd.DataFrame]:
    raw = pd.read_csv(path)
    if list(raw.columns) != ["Time", "Power consumption"]:
        raise ValueError(f"unexpected factory columns in {path}: {list(raw.columns)}")
    timestamps = pd.to_datetime(raw["Time"], errors="coerce")
    values = pd.to_numeric(raw["Power consumption"], errors="coerce")
    duplicate = timestamps.notna() & timestamps.duplicated(keep="first")
    negative = values < 0.0
    flags = []
    for mask, flag_type in (
        (timestamps.isna(), "unparseable_timestamp"),
        (duplicate, "duplicate_timestamp_after_first"),
        (values.isna(), "nonnumeric_load"),
        (negative, "negative_load"),
    ):
        selected = raw.loc[mask, ["Time", "Power consumption"]].copy()
        for source_index, row in selected.iterrows():
            flags.append(
                {
                    "factory": factory,
                    "source_row": int(source_index) + 2,
                    "timestamp_raw": str(row["Time"]),
                    "load_raw": str(row["Power consumption"]),
                    "flag_type": flag_type,
                    "action_main": "invalid_or_drop_duplicate_after_first",
                }
            )
    valid = pd.DataFrame({"timestamp": timestamps, "load_kw": values})
    valid = valid[timestamps.notna() & ~duplicate & values.notna() & ~negative]
    series = valid.set_index("timestamp")["load_kw"].sort_index().reindex(_full_index(config))
    finite = series.dropna()
    q1, q3 = finite.quantile([0.25, 0.75])
    iqr_upper = float(q3 + 1.5 * (q3 - q1))
    exploratory = finite[finite > iqr_upper]
    for timestamp, value in exploratory.items():
        flags.append(
            {
                "factory": factory,
                "source_row": None,
                "timestamp_raw": timestamp.isoformat(),
                "load_raw": str(float(value)),
                "flag_type": "provider_script_iqr_upper_fence",
                "action_main": "retain_exploratory_flag_only",
            }
        )
    quality = {
        "factory": factory,
        "raw_rows": len(raw),
        "unparseable_timestamps": int(timestamps.isna().sum()),
        "duplicate_timestamps_after_first": int(duplicate.sum()),
        "nonnumeric_load_rows": int(values.isna().sum()),
        "negative_load_rows": int(negative.sum()),
        "valid_unique_minutes": int(series.notna().sum()),
        "expected_minutes": len(series),
        "minute_completeness": float(series.notna().mean()),
        "minimum_kw": float(finite.min()),
        "maximum_kw": float(finite.max()),
        "iqr_upper_fence_kw": iqr_upper,
        "exploratory_iqr_flag_count": len(exploratory),
    }
    return series, quality, pd.DataFrame(flags, columns=FLAG_COLUMNS)


def dr_exclusion_mask(
    index: pd.DatetimeIndex,
    factory: str,
    intervals: pd.DataFrame,
) -> np.ndarray:
    mask = np.zeros(len(index), dtype=bool)
    subset = intervals[intervals["factory"] == factory]
    for row in subset.itertuples(index=False):
        mask |= (index >= row.event_start) & (index < row.exclusion_end)
    return mask


def aggregate_complete_windows(
    series: pd.Series,
    window_minutes: int,
    sliding: bool,
) -> pd.DataFrame:
    if sliding:
        rolling = series.rolling(window=window_minutes, min_periods=window_minutes)
        frame = pd.DataFrame({"load_kw": rolling.mean(), "valid_minutes": rolling.count()})
    else:
        aggregate = series.resample(f"{window_minutes}min", origin="start_day").agg(["mean", "count"])
        frame = aggregate.rename(columns={"mean": "load_kw", "count": "valid_minutes"})
    frame["load_kw"] = frame["load_kw"].where(frame["valid_minutes"] == window_minutes)
    frame.index.name = "window_timestamp"
    return frame.reset_index()


def reconstruct_factory(
    path: str | Path,
    factory: str,
    config: dict[str, Any],
    intervals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    series, quality, anomaly_flags = load_factory_minutes(path, factory, config)
    exclusion = dr_exclusion_mask(series.index, factory, intervals)
    variants = {
        "all_observations": series,
        "natural_tail": series.mask(exclusion),
    }
    fixed_parts = []
    sliding_parts = []
    for variant, values in variants.items():
        for sliding, parts, measurement in (
            (False, fixed_parts, "fixed_nonoverlapping_15min"),
            (True, sliding_parts, "sliding_15min"),
        ):
            frame = aggregate_complete_windows(values, int(config["main_window_minutes"]), sliding)
            frame.insert(0, "factory", factory)
            frame["variant"] = variant
            frame["measurement"] = measurement
            parts.append(frame)
    quality["dr_event_count"] = int((intervals["factory"] == factory).sum())
    quality["dr_buffer_excluded_minutes"] = int(exclusion.sum())
    return (
        pd.concat(fixed_parts, ignore_index=True),
        pd.concat(sliding_parts, ignore_index=True),
        quality,
        anomaly_flags,
    )


def monthly_eligibility(windows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    frame = windows.copy()
    frame["month"] = frame["window_timestamp"].dt.to_period("M").astype(str)
    rows = []
    for keys, group in frame.groupby(["factory", "variant", "measurement", "month"]):
        valid = int(group["load_kw"].notna().sum())
        expected = len(group)
        fraction = valid / expected if expected else float("nan")
        rows.append(
            {
                "factory": keys[0],
                "variant": keys[1],
                "measurement": keys[2],
                "month": keys[3],
                "expected_windows": expected,
                "valid_windows": valid,
                "valid_fraction": fraction,
                "eligible_95": fraction >= float(config["month_eligibility_main"]),
                "eligible_90": fraction >= float(config["month_eligibility_robustness"]),
                "monthly_max_kw": float(group["load_kw"].max()) if valid else None,
            }
        )
    return pd.DataFrame(rows)
