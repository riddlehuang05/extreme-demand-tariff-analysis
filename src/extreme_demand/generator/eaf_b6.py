"""Minute-level EAF-80SS-B6 simulation with explicit evidence boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
from scipy import signal

from extreme_demand.aggregation import fixed_nonoverlapping_average, sliding_average


@dataclass(frozen=True)
class GeneratorConfig:
    scenario_id: str
    resolution_minutes: int
    billing_minutes: int
    active_power_max_mw: float
    annual_heats_target: int
    ttt_mean: float
    ttt_sd: float
    ttt_lower: int
    ttt_upper: int
    idle_shape: float
    idle_mean: float
    stage_names: tuple[str, ...]
    stage_duration_center: np.ndarray
    stage_concentration: float
    stage_power_ratio: np.ndarray
    heat_energy_mean: float
    heat_energy_cv: float
    heat_energy_lower: float
    heat_energy_upper: float
    ar_phi: float
    ar_sd_fraction: float
    physical_heat_energy_upper: float
    background_mean: float
    background_amplitude: float
    background_peak_hour: float
    background_phi: float
    background_innovation_sd: float
    background_lower: float
    background_upper: float
    rare_cluster_rate: float
    rare_headroom_fraction: float
    rare_duration: int
    consecutive_heat_probability: float
    rare_truth_family: str
    rare_power_cap_mw: float
    days_per_month: int
    days_per_year: int
    beta_alpha: float
    beta_beta: float
    lognormal_mean_fraction: float
    lognormal_cv: float
    gpd_xi: float
    gpd_mean_fraction: float

    @classmethod
    def from_yaml(cls, generator_path: str | Path, assumptions_path: str | Path) -> "GeneratorConfig":
        with Path(generator_path).open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)["generator"]
        with Path(assumptions_path).open("r", encoding="utf-8") as handle:
            assumptions = yaml.safe_load(handle)["pilot_author_assumptions"]
        furnace = document["furnace"]
        ttt = furnace["ttt"]
        idle = furnace["idle"]
        stages = furnace["stages"]
        energy = furnace["heat_energy"]
        noise = furnace["within_stage_noise"]
        background = document["background"]
        rare = document["rare_process_disturbance"]
        severity = assumptions["rare_severity"]
        calendar = assumptions["synthetic_calendar"]
        config = cls(
            scenario_id=str(document["scenario_id"]),
            resolution_minutes=int(document["resolution_minutes"]),
            billing_minutes=int(document["billing_resolution_minutes"]),
            active_power_max_mw=float(furnace["active_power_max_mw"]),
            annual_heats_target=int(furnace["annual_heats_target"]),
            ttt_mean=float(ttt["mean_min"]),
            ttt_sd=float(ttt["sd_min"]),
            ttt_lower=int(ttt["lower_min"]),
            ttt_upper=int(ttt["upper_min"]),
            idle_shape=float(idle["shape"]),
            idle_mean=float(idle["mean_min"]),
            stage_names=tuple(stages["names"]),
            stage_duration_center=np.asarray(stages["duration_center"], dtype=float),
            stage_concentration=float(stages["duration_dirichlet_concentration"]),
            stage_power_ratio=np.asarray(stages["raw_power_ratio"], dtype=float),
            heat_energy_mean=float(energy["mean_mwh"]),
            heat_energy_cv=float(energy["cv"]),
            heat_energy_lower=float(energy["lower_mwh"]),
            heat_energy_upper=float(energy["upper_mwh"]),
            ar_phi=float(noise["phi"]),
            ar_sd_fraction=float(noise["sd_fraction_of_stage_power"]),
            physical_heat_energy_upper=float(furnace["physical_checks"]["heat_energy_upper_mwh"]),
            background_mean=float(background["mean_mw"]),
            background_amplitude=float(background["diurnal_amplitude_mw"]),
            background_peak_hour=float(background["peak_hour_local"]),
            background_phi=float(background["ar_phi"]),
            background_innovation_sd=float(background["innovation_sd_mw"]),
            background_lower=float(background["lower_mw"]),
            background_upper=float(background["upper_mw"]),
            rare_cluster_rate=float(rare["cluster_rate_per_month"]),
            rare_headroom_fraction=float(rare["amplitude_headroom_fraction"]),
            rare_duration=int(rare["duration_minutes"]),
            consecutive_heat_probability=float(rare["consecutive_heat_probability"]),
            rare_truth_family=str(rare["truth_family"]),
            rare_power_cap_mw=float(rare["furnace_power_cap_mw"]),
            days_per_month=int(calendar["days_per_month"]),
            days_per_year=int(calendar["days_per_year_for_annualization"]),
            beta_alpha=float(severity["bounded_beta"]["alpha"]),
            beta_beta=float(severity["bounded_beta"]["beta"]),
            lognormal_mean_fraction=float(severity["lognormal"]["mean_fraction"]),
            lognormal_cv=float(severity["lognormal"]["cv"]),
            gpd_xi=float(severity["gpd"]["shape_xi"]),
            gpd_mean_fraction=float(severity["gpd"]["mean_fraction"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        n_stages = len(self.stage_names)
        if n_stages != 6 or self.stage_duration_center.size != n_stages or self.stage_power_ratio.size != n_stages:
            raise ValueError("B6 generator must have six aligned stages")
        if self.resolution_minutes != 1 or self.billing_minutes != 15:
            raise ValueError("Gate 2 requires one-minute paths and 15-minute billing windows")
        if not np.isclose(self.stage_duration_center.sum(), 1.0, atol=1e-12):
            raise ValueError("stage duration centers must sum to one")
        if np.any(self.stage_duration_center <= 0.0) or np.any(self.stage_power_ratio <= 0.0):
            raise ValueError("stage centers and raw power ratios must be positive")
        if not (0.0 <= self.ar_phi < 1.0 and 0.0 <= self.background_phi < 1.0):
            raise ValueError("AR coefficients must lie in [0, 1)")
        if not self.ttt_lower <= self.ttt_mean <= self.ttt_upper:
            raise ValueError("TTT mean must lie inside the truncation bounds")
        if self.ttt_sd <= 0.0 or self.idle_shape <= 0.0 or self.idle_mean <= 0.0:
            raise ValueError("TTT and idle distribution parameters must be positive")
        if not (0.0 < self.heat_energy_lower < self.heat_energy_mean < self.heat_energy_upper):
            raise ValueError("invalid heat-energy truncation parameters")
        if self.active_power_max_mw <= 0.0 or self.rare_power_cap_mw != self.active_power_max_mw:
            raise ValueError("rare-event cap must equal the frozen furnace active-power cap")
        if not (0.0 <= self.background_lower < self.background_mean < self.background_upper):
            raise ValueError("invalid background bounds")
        if not 0.0 <= self.consecutive_heat_probability <= 1.0:
            raise ValueError("consecutive heat probability must lie in [0, 1]")
        if self.rare_truth_family not in {"bounded_beta", "lognormal", "gpd"}:
            raise ValueError("unsupported rare truth family")
        if self.days_per_month <= 0 or self.days_per_year <= 0:
            raise ValueError("synthetic calendar lengths must be positive")
        if self.beta_alpha <= 0.0 or self.beta_beta <= 0.0:
            raise ValueError("Beta severity parameters must be positive")
        if not 0.0 < self.gpd_xi < 1.0:
            raise ValueError("pilot GPD shape must lie in (0, 1) for a finite positive mean")
        if self.heat_energy_upper > self.physical_heat_energy_upper:
            raise ValueError("sampled heat-energy upper bound exceeds the physical-check bound")
        expected_cycle = self.ttt_mean + self.idle_mean
        implied_heats = self.days_per_year * 24.0 * 60.0 / expected_cycle
        if abs(implied_heats - self.annual_heats_target) / self.annual_heats_target > 0.001:
            raise ValueError("TTT and idle means do not reproduce the annual heat target")

    @property
    def minutes_per_month(self) -> int:
        return self.days_per_month * 24 * 60


@dataclass
class HeatRecord:
    heat_id: int
    start_minute: int
    duration_min: int
    stage_durations: np.ndarray
    stage_labels: np.ndarray
    power_mw: np.ndarray
    sampled_energy_mwh: float
    realized_energy_mwh: float
    rare_event_flag: bool
    rare_cluster_id: int | None
    rare_family: str
    rare_severity: float
    rare_added_energy_mwh: float
    rare_start_offset_min: int | None
    rare_duration_min: int
    rare_state: str
    rare_energy_cap_applied: bool


@dataclass
class MonthRecord:
    month_id: int
    eaf_mw: np.ndarray
    background_mw: np.ndarray
    poc_mw: np.ndarray
    fixed15_mw: np.ndarray
    sliding15_mw: np.ndarray
    heats: list[HeatRecord]
    rare_cluster_count: int


def draw_truncated_normal(rng: np.random.Generator, mean: float, sd: float, lower: float, upper: float) -> float:
    while True:
        value = float(rng.normal(mean, sd))
        if lower <= value <= upper:
            return value


def draw_truncated_lognormal(
    rng: np.random.Generator, mean: float, cv: float, lower: float, upper: float
) -> float:
    sigma = np.sqrt(np.log1p(cv**2))
    location = np.log(mean) - 0.5 * sigma**2
    while True:
        value = float(rng.lognormal(location, sigma))
        if lower <= value <= upper:
            return value


def allocate_stage_minutes(total_minutes: int, shares: np.ndarray) -> np.ndarray:
    shares = np.asarray(shares, dtype=float)
    if shares.ndim != 1 or shares.size == 0 or not np.all(np.isfinite(shares)):
        raise ValueError("shares must be a finite one-dimensional array")
    if np.any(shares <= 0.0) or not np.isclose(shares.sum(), 1.0, atol=1e-10):
        raise ValueError("shares must be positive and sum to one")
    if total_minutes < shares.size:
        raise ValueError("total duration is too short to allocate every stage")
    raw = shares * total_minutes
    durations = np.maximum(np.floor(raw).astype(int), 1)
    while durations.sum() < total_minutes:
        index = int(np.argmax(raw - durations))
        durations[index] += 1
    while durations.sum() > total_minutes:
        removable = np.where(durations > 1, durations - raw, -np.inf)
        index = int(np.argmax(removable))
        durations[index] -= 1
    return durations


def scale_profile_to_energy(power_mw: np.ndarray, target_mwh: float, cap_mw: float) -> np.ndarray:
    profile = np.asarray(power_mw, dtype=float)
    if profile.ndim != 1 or profile.size == 0 or not np.all(np.isfinite(profile)):
        raise ValueError("power profile must be finite, one-dimensional, and non-empty")
    if np.any(profile <= 0.0) or target_mwh <= 0.0 or cap_mw <= 0.0:
        raise ValueError("profile, target energy, and cap must be positive")
    target_sum = target_mwh * 60.0
    if target_sum > cap_mw * profile.size + 1e-10:
        raise ValueError("target heat energy is infeasible under the power cap")
    scaled = np.zeros_like(profile)
    active = np.ones(profile.size, dtype=bool)
    remaining = target_sum
    while np.any(active):
        factor = remaining / profile[active].sum()
        proposed = factor * profile[active]
        newly_capped_local = proposed > cap_mw
        if not np.any(newly_capped_local):
            scaled[active] = proposed
            break
        active_indices = np.flatnonzero(active)
        capped_indices = active_indices[newly_capped_local]
        scaled[capped_indices] = cap_mw
        active[capped_indices] = False
        remaining -= cap_mw * capped_indices.size
    return scaled


def draw_rare_severity(cfg: GeneratorConfig, rng: np.random.Generator, family: str) -> float:
    if family == "bounded_beta":
        return float(rng.beta(cfg.beta_alpha, cfg.beta_beta))
    if family == "lognormal":
        sigma = np.sqrt(np.log1p(cfg.lognormal_cv**2))
        location = np.log(cfg.lognormal_mean_fraction) - 0.5 * sigma**2
        return float(rng.lognormal(location, sigma))
    if family == "gpd":
        scale = cfg.gpd_mean_fraction * (1.0 - cfg.gpd_xi)
        return float(rng.pareto(1.0 / cfg.gpd_xi) * scale / cfg.gpd_xi)
    raise ValueError(f"unsupported rare severity family: {family}")


def generate_heat(
    heat_id: int,
    start_minute: int,
    duration_min: int,
    cfg: GeneratorConfig,
    stage_rng: np.random.Generator,
    energy_rng: np.random.Generator,
    rare_rng: np.random.Generator,
    rare_cluster_id: int | None = None,
    rare_family: str | None = None,
) -> HeatRecord:
    shares = stage_rng.dirichlet(cfg.stage_duration_center * cfg.stage_concentration)
    stage_durations = allocate_stage_minutes(duration_min, shares)
    stage_indices = np.repeat(np.arange(len(cfg.stage_names)), stage_durations)
    nominal = cfg.active_power_max_mw * cfg.stage_power_ratio[stage_indices]
    innovations = stage_rng.normal(0.0, cfg.ar_sd_fraction, size=duration_min)
    deviations = signal.lfilter([1.0], [1.0, -cfg.ar_phi], innovations)
    raw_power = np.maximum(nominal * (1.0 + deviations), 0.01)
    sampled_energy = draw_truncated_lognormal(
        energy_rng,
        cfg.heat_energy_mean,
        cfg.heat_energy_cv,
        cfg.heat_energy_lower,
        cfg.heat_energy_upper,
    )
    power = scale_profile_to_energy(raw_power, sampled_energy, cfg.active_power_max_mw)
    base_energy = float(power.sum() / 60.0)
    severity = 0.0
    rare_added_energy = 0.0
    rare_start_offset: int | None = None
    rare_duration = 0
    rare_state = "none"
    rare_energy_cap_applied = False
    family = rare_family or cfg.rare_truth_family
    if rare_cluster_id is not None:
        severity = draw_rare_severity(cfg, rare_rng, family)
        melting_index = cfg.stage_names.index("melting")
        eligible = np.flatnonzero(stage_indices == melting_index)
        event_length = min(cfg.rare_duration, eligible.size)
        max_offset = eligible.size - event_length
        offset = int(rare_rng.integers(0, max_offset + 1)) if max_offset > 0 else 0
        event_indices = eligible[offset : offset + event_length]
        rare_start_offset = int(event_indices[0])
        rare_duration = int(event_length)
        rare_state = "melting"
        headroom = cfg.rare_power_cap_mw - power[event_indices]
        addition = cfg.rare_headroom_fraction * severity * headroom
        proposed_added_energy = float(addition.sum() / 60.0)
        allowed_added_energy = max(0.0, cfg.physical_heat_energy_upper - base_energy)
        if proposed_added_energy > allowed_added_energy and proposed_added_energy > 0.0:
            addition *= allowed_added_energy / proposed_added_energy
            rare_energy_cap_applied = True
        power[event_indices] = np.minimum(power[event_indices] + addition, cfg.rare_power_cap_mw)
        rare_added_energy = float(power.sum() / 60.0 - base_energy)
    realized_energy = float(power.sum() / 60.0)
    return HeatRecord(
        heat_id=heat_id,
        start_minute=start_minute,
        duration_min=duration_min,
        stage_durations=stage_durations,
        stage_labels=stage_indices,
        power_mw=power,
        sampled_energy_mwh=sampled_energy,
        realized_energy_mwh=realized_energy,
        rare_event_flag=rare_cluster_id is not None,
        rare_cluster_id=rare_cluster_id,
        rare_family=family if rare_cluster_id is not None else "none",
        rare_severity=severity,
        rare_added_energy_mwh=rare_added_energy,
        rare_start_offset_min=rare_start_offset,
        rare_duration_min=rare_duration,
        rare_state=rare_state,
        rare_energy_cap_applied=rare_energy_cap_applied,
    )


def _schedule_month(cfg: GeneratorConfig, calendar_rng: np.random.Generator) -> list[tuple[int, int, int]]:
    schedule: list[tuple[int, int, int]] = []
    cursor = 0
    heat_id = 0
    while True:
        idle = max(0, int(round(calendar_rng.gamma(cfg.idle_shape, cfg.idle_mean / cfg.idle_shape))))
        start = cursor + idle
        if start >= cfg.minutes_per_month:
            break
        duration = int(round(draw_truncated_normal(calendar_rng, cfg.ttt_mean, cfg.ttt_sd, cfg.ttt_lower, cfg.ttt_upper)))
        schedule.append((heat_id, start, duration))
        heat_id += 1
        cursor = start + duration
    return schedule


def _rare_assignments(
    heat_count: int, cfg: GeneratorConfig, rare_rng: np.random.Generator
) -> tuple[dict[int, int], int]:
    cluster_count = int(rare_rng.poisson(cfg.rare_cluster_rate))
    if heat_count == 0 or cluster_count == 0:
        return {}, 0
    anchors = rare_rng.choice(heat_count, size=min(cluster_count, heat_count), replace=False)
    assignments: dict[int, int] = {}
    for cluster_id, anchor in enumerate(np.sort(anchors), start=1):
        anchor_int = int(anchor)
        assignments[anchor_int] = cluster_id
        if anchor_int + 1 < heat_count and rare_rng.random() < cfg.consecutive_heat_probability:
            assignments.setdefault(anchor_int + 1, cluster_id)
    return assignments, min(cluster_count, heat_count)


def generate_background(cfg: GeneratorConfig, rng: np.random.Generator, size: int) -> np.ndarray:
    minute = np.arange(size)
    hour = (minute % (24 * 60)) / 60.0
    diurnal = cfg.background_amplitude * np.cos(
        2.0 * np.pi * (hour - cfg.background_peak_hour) / 24.0
    )
    stationary_sd = cfg.background_innovation_sd / np.sqrt(1.0 - cfg.background_phi**2)
    initial = float(rng.normal(0.0, stationary_sd))
    innovations = rng.normal(0.0, cfg.background_innovation_sd, size=size)
    ar_component, _ = signal.lfilter(
        [1.0], [1.0, -cfg.background_phi], innovations, zi=[cfg.background_phi * initial]
    )
    return np.clip(
        cfg.background_mean + diurnal + ar_component,
        cfg.background_lower,
        cfg.background_upper,
    )


def simulate_month(
    month_id: int,
    cfg: GeneratorConfig,
    rngs: dict[str, np.random.Generator],
) -> MonthRecord:
    schedule = _schedule_month(cfg, rngs["calendar"])
    rare_assignments, cluster_count = _rare_assignments(len(schedule), cfg, rngs["rare_events"])
    eaf = np.zeros(cfg.minutes_per_month, dtype=float)
    heats: list[HeatRecord] = []
    for heat_id, start, duration in schedule:
        heat = generate_heat(
            heat_id=heat_id,
            start_minute=start,
            duration_min=duration,
            cfg=cfg,
            stage_rng=rngs["stages"],
            energy_rng=rngs["energy"],
            rare_rng=rngs["rare_events"],
            rare_cluster_id=rare_assignments.get(heat_id),
        )
        end = min(start + duration, cfg.minutes_per_month)
        eaf[start:end] = heat.power_mw[: end - start]
        heats.append(heat)
    background = generate_background(cfg, rngs["background"], cfg.minutes_per_month)
    poc = eaf + background
    return MonthRecord(
        month_id=month_id,
        eaf_mw=eaf,
        background_mw=background,
        poc_mw=poc,
        fixed15_mw=fixed_nonoverlapping_average(poc, cfg.billing_minutes),
        sliding15_mw=sliding_average(poc, cfg.billing_minutes),
        heats=heats,
        rare_cluster_count=cluster_count,
    )
