#!/usr/bin/env python3
"""Four-fold CV-CRPS primary refit over the hash-verified history release.

The script requires saved monthly/event histories and never changes or silently
regenerates them. It refits TAIL/GEV without altering the source engine. The
TAIL threshold is selected by four-fold month-level CRPS, and its selected
full-history GPD fit receives a 500-resample shape diagnostic. The default
policy is the nonnegative/no-artificial-upper-bound analysis.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import integrate, special, stats
from primary_cv_crps import select_threshold as select_threshold_cv_crps

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parent
DEFAULT_BASE = PACKAGE.parent / "GITHUB_UPLOAD_FINAL_20260924" / "extreme-demand-tariff-analysis"
DEFAULT_DATA = PACKAGE / "data" / "primary_histories_offline_20261001"
_ENGINES: dict[str, Any] = {}
_HASH_CACHE: dict[str, str] = {}


def load_engine(base: Path):
    key = str(base.resolve())
    if key not in _ENGINES:
        script = base / "scripts" / "run_revision_experiment.py"
        if not script.is_file():
            raise FileNotFoundError(script)
        sys.path.insert(0, str(base / "src"))
        spec = importlib.util.spec_from_file_location("support_policy_source_engine", script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load source engine: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _ENGINES[key] = module
    return _ENGINES[key]


def safe(x: Any):
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return safe(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        y = float(x)
        return y if math.isfinite(y) else None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x if x is None or isinstance(x, (str, int)) else str(x)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(rows: list[dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


class NonnegativeGEV:
    """Finite-mean GEV conditioned on X>=0, with no upper truncation."""
    def __init__(self, shape_c: float, loc: float, scale: float):
        self.shape_c, self.xi = float(shape_c), -float(shape_c)
        self.loc, self.scale = float(loc), float(scale)
        if self.xi >= 1 or self.scale <= 0 or not np.isfinite([self.xi, self.loc, self.scale]).all():
            raise ValueError("invalid/non-finite-mean GEV parameters")
        self.f0 = float(stats.genextreme.cdf(0, self.shape_c, loc=self.loc, scale=self.scale))
        self.a = float(stats.genextreme.sf(0, self.shape_c, loc=self.loc, scale=self.scale))
        if self.a <= 0 or not math.isfinite(self.a):
            raise ValueError("no numerically resolvable nonnegative GEV mass")
        self.raw_mean = (self.loc + self.scale * float(np.euler_gamma) if self.xi == 0.0
                         else self.loc + self.scale * math.expm1(float(special.gammaln(1-self.xi)))/self.xi)
        self.lower = self.loc-self.scale/self.xi if self.xi > 0 else -math.inf
        self.upper = self.loc-self.scale/self.xi if self.xi < 0 else math.inf
        self.max_quad_abs_error = 0.0

    def cdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        raw = stats.genextreme.cdf(x, self.shape_c, loc=self.loc, scale=self.scale)
        return float(np.clip((raw-self.f0)/self.a, 0, 1))

    def ppf(self, p: float) -> float:
        raw_p = self.f0 + float(p)*self.a
        return max(0.0, float(stats.genextreme.ppf(raw_p, self.shape_c, loc=self.loc, scale=self.scale)))

    def raw_stop_loss(self, t: float) -> float:
        t = float(t)
        if t <= self.lower:
            return max(0.0, self.raw_mean-t)
        if t >= self.upper:
            return 0.0
        if self.xi == 0.0:
            logv = -(t-self.loc)/self.scale
        else:
            logbase=math.log1p(self.xi*(t-self.loc)/self.scale)
            base=math.exp(logbase) if logbase>-745 else 0.0
            if base <= 0:
                return max(0.0, self.raw_mean-t) if self.xi > 0 else 0.0
            logv = -logbase/self.xi
        if logv > 700:
            return max(0.0, self.raw_mean-t)
        v = math.exp(logv)
        if v == 0:
            return 0.0
        if self.xi == 0.0:
            # Exact Gumbel integral: H(t)=scale*(log(v)+EulerGamma+E1(v)).
            if v < 1e-4:
                term, series, k = v, 0.0, 1
                while abs(term) > 1e-18 and k < 100:
                    term = ((-1.0)**(k+1))*v**k/(k*math.factorial(k))
                    series += term
                    k += 1
                return max(0.0, self.scale*series)
            e1 = float(special.exp1(v)) if v < 745 else 0.0
            return max(0.0, self.scale*(logv+float(np.euler_gamma)+e1))
        if abs(self.xi) < 1e-5 and v <= 50:
            # Near xi=0 the incomplete-gamma closed form subtracts nearly equal
            # quantities. Integrate directly in the Exp(1) variable and retain
            # QUADPACK's absolute error as an auditable numerical diagnostic.
            def integrand(z):
                x = self.loc+self.scale*math.expm1(-self.xi*math.log(z))/self.xi
                return max(0.0, x-t)*math.exp(-z)
            value, error = integrate.quad(integrand, 0.0, v, epsabs=1e-10*self.scale,
                                          epsrel=2e-11, limit=300)
            self.max_quad_abs_error = max(self.max_quad_abs_error, float(error))
            return max(0.0, float(value))
        # Closed form from V~Exp(1):
        # (loc-t) A + scale/xi * [Gamma(1-xi) P(1-xi,v)-A], A=1-exp(-v).
        A = -math.expm1(-v)
        shape = 1.0-self.xi
        gamma_lower = float(special.gamma(shape)*special.gammainc(shape, v))
        if v > 50:
            gamma_minus_one = math.expm1(float(special.gammaln(shape)))
            bracket = gamma_minus_one
        else:
            bracket = gamma_lower-A
        value = (self.loc-t)*A+self.scale/self.xi*bracket
        return max(0.0, float(value))

    def stop_loss(self, threshold: float) -> float:
        t=float(threshold)
        return self.mean()-t if t < 0 else self.raw_stop_loss(t)/self.a

    def mean(self) -> float:
        return self.stop_loss(0.0)


def self_check():
    # Independent truncated-mean formula: incomplete gamma (xi != 0), direct
    # Exp(1) integral for the Gumbel limit.
    rows = []
    for xi, loc, scale in [(-0.9, 3.5, 1.0), (-0.4, 2.2, 1.0),
                           (0.0, 2.0, 1.0), (0.35, 1.6, 1.0), (0.9, 5.0, 1.0)]:
        d = NonnegativeGEV(-xi, loc, scale)
        if d.f0 > 0.01:
            raise AssertionError("self-check violates the 1% lower-support gate")
        base0=1+xi*(0-loc)/scale
        if xi>0 and base0<=0:  # all raw support is already nonnegative
            truth=d.raw_mean
        else:
            logv0 = -(0-loc)/scale if xi == 0 else -math.log(base0)/xi
            v0 = math.exp(logv0)
            a = -math.expm1(-v0)
            if xi == 0:
                num = integrate.quad(lambda v: (loc-scale*math.log(v))*math.exp(-v), 0, v0,
                                     epsabs=1e-10, epsrel=1e-10, limit=300)[0]
                truth = num/a
            else:
                shape = 1-xi
                low_gamma = float(special.gamma(shape)*special.gammainc(shape, v0))
                truth = loc+scale/xi*(low_gamma/a-1)
        value = d.mean()
        err = abs(value-truth)
        sl_checks=[]
        for t in (0.0,0.25*loc,0.75*loc,1.2*loc):
            base=1+xi*(t-loc)/scale
            if xi and base<=0:
                continue
            logv=-(t-loc)/scale if xi==0 else -math.log(base)/xi
            if logv>700:
                continue
            vt=math.exp(logv)
            if xi==0:
                integrand=lambda z:(loc-scale*math.log(z)-t)*math.exp(-z)
            else:
                integrand=lambda z:(loc+scale*math.expm1(-xi*math.log(z))/xi-t)*math.exp(-z)
            raw_ref=integrate.quad(integrand,0.0,vt,epsabs=1e-10*scale,epsrel=2e-10,limit=400)[0]
            ref=raw_ref/d.a; got=d.stop_loss(t); slerr=abs(got-ref)
            sl_checks.append({"threshold":t,"calculated":got,"adaptive_reference":ref,"abs_error":slerr})
            if slerr>3e-7*max(1,abs(ref)):
                raise AssertionError(f"stop-loss adaptive check failed: xi={xi}, threshold={t}, error={slerr}")
        q_checks=[]
        for p in (0.25,0.75,0.99):
            q=d.ppf(p); back=d.cdf(q); qerr=abs(back-p)
            q_checks.append({"p":p,"q":q,"cdf_ppf_abs_error":qerr})
            if qerr>2e-9:
                raise AssertionError(f"GEV quantile inversion failed: xi={xi}, p={p}, error={qerr}")
        rows.append({"xi": xi, "mean_stable_integral": value, "mean_independent": truth,
                     "abs_error": err, "raw_negative_mass": d.f0,"stoploss_checks":sl_checks,
                     "quantile_checks":q_checks,"max_adaptive_quad_abs_error":d.max_quad_abs_error})
        if not math.isfinite(value) or err > 2e-7*max(1, abs(truth)):
            raise AssertionError(f"GEV integral self-check failed: {rows[-1]}")
    return {"status": "PASS", "cases": rows}


def fit_tail(engine, monthly, events, settings, rep, kind, bound):
    if kind == "no_upper":
        g = engine.fit_process_tail.__globals__
        processed = g["_process_events"](events, True)
        selected_q, fold_rows = select_threshold_cv_crps(monthly, processed)
        values = processed.cluster_max_kw.to_numpy(dtype=float)
        threshold = float(np.quantile(values, selected_q))
        excesses = values[values > threshold] - threshold
        if excesses.size == 0:
            raise RuntimeError(f"selected CV threshold has no exceedances for history {rep}")
        xi, _, sigma = map(float, stats.genpareto.fit(excesses, floc=0.0))
        bootstrap_rng = engine.seed_rng(settings["seed_root"], 2, 0, rep, 20, 2)
        shape_width = float(g["_shape_ci_width"](excesses, 500, bootstrap_rng))
        months = np.sort(monthly.month_id.unique())
        counts = (processed.assign(exceed=processed.cluster_max_kw > threshold)
                  .groupby("month_id").exceed.sum().reindex(months, fill_value=0).to_numpy(int))
        count_model, count_mean, count_var, nb_size, nb_prob = g["_count_parameters"](counts)
        endpoint = threshold - sigma / xi if xi < 0 else None
        full_values = processed.cluster_max_kw.to_numpy(dtype=float)
        reasons = []
        if excesses.size < int(settings["tail"]["minimum_exceedances_main"]):
            reasons.append("fewer_than_minimum_exceedances")
        if shape_width > float(settings["tail"]["shape_ci_width_gate"]):
            reasons.append("shape_ci_width")
        if xi >= 1.0 or not np.isfinite([xi, sigma]).all() or sigma <= 0.0:
            reasons.append("invalid_finite_mean_fit")
        if endpoint is not None and endpoint < full_values.max() - 1e-9:
            reasons.append("finite_endpoint_observed_conflict")
        fold_scores = {float(q): float(np.mean([row.crps for row in fold_rows
                                               if row.quantile == float(q)
                                               and row.status == "PASS"]))
                       for q in settings["tail"]["threshold_quantiles"]}
        selected_score = fold_scores[float(selected_q)]
        candidate = {
            "quantile": float(selected_q), "threshold": threshold, "xi": xi,
            "sigma": sigma, "exceedances": int(excesses.size),
            "shape_ci_width": shape_width, "endpoint_kw": endpoint,
            "ks_pvalue": float("nan"), "count_model": count_model,
            "count_mean": count_mean, "count_variance": count_var,
            "nb_size": nb_size, "nb_probability": nb_prob,
            "validation_log_score": -selected_score,
            "validation_crps": selected_score,
            "selection_method": "four_fold_month_level_cv_crps",
            "valid_main": not reasons, "rejection_reasons": reasons,
            "fold_scores": [row.__dict__ for row in fold_rows],
            "candidate_cv_mean_crps": fold_scores,
        }
        old_process = engine.fit_process_tail
        def selected_candidate(*_args, **_kwargs):
            return g["FitResult"]("TAIL-JOINT", np.array([threshold]), "PASS",
                                  diagnostics={"candidate_diagnostics": [candidate]})
        engine.fit_process_tail = selected_candidate
        try:
            d, diag = engine.fit_tail_exact(
                monthly, events, settings, "full",
                engine.seed_rng(settings["seed_root"], 2, 0, rep, 20, 1),
                engine.seed_rng(settings["seed_root"], 2, 0, rep, 20, 2),
                float(np.finfo(float).max))
        finally:
            engine.fit_process_tail = old_process
        diag["selection_method"] = "four_fold_month_level_cv_crps"
        diag["selected_candidate_cv_mean_crps"] = selected_score
        diag["selection_fit_months"] = 18
        diag["selection_validation_months"] = 6
        diag["selection_folds"] = 4
        diag["selection_total_held_out_months"] = 24
        diag["shape_bootstrap_replicates"] = 500
        diag["full_fit_shape_ci_width"] = shape_width
        return d, diag, {"selection_method": "four_fold_month_level_cv_crps",
                         "selected_quantile": float(selected_q),
                         "fold_scores": [row.__dict__ for row in fold_rows],
                         "candidate_cv_mean_crps": fold_scores,
                         "full_fit_shape_ci_width_500": shape_width,
                         "full_fit_candidate_rejections": reasons}
    raise ValueError(f"unsupported TAIL policy kind: {kind}")


def load_manifest(path):
    for i in range(4):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if i == 3:
                raise
            time.sleep(.2)


def load_raw(data: Path, manifest: dict, ids: list[int]):
    need = set(ids)
    pairs = {}
    for chunk in manifest.get("chunks", []):
        rec = chunk.get("history_record") or chunk
        mp = rec.get("monthly_path") or rec.get("monthly_file")
        ep = rec.get("event_path") or rec.get("event_file")
        if chunk.get("status", "PASS") != "PASS" or not mp or not ep:
            continue
        wanted = {r for r in ids if int(chunk.get("start", 0)) <= r < int(chunk.get("end", 0))}
        if wanted:
            base=Path(manifest.get("_manifest_dir",data))
            mp=Path(mp); ep=Path(ep)
            if not mp.is_absolute(): mp=base/mp
            if not ep.is_absolute(): ep=base/ep
            pairs[(str(mp),str(ep))]=(wanted,rec.get("monthly_sha256"),rec.get("event_sha256"))
    # If a finalizer has removed intermediate chunks, use the merged raw tables
    # for uncovered IDs instead of silently regenerating an already saved run.
    pairs={paths:record for paths,record in pairs.items()
           if Path(paths[0]).is_file() and Path(paths[1]).is_file()}
    covered=set().union(*(v[0] for v in pairs.values())) if pairs else set()
    missing=need-covered
    outputs=manifest.get("outputs", {})
    mp, ep=outputs.get("monthly_maxima"),outputs.get("event_clusters")
    if missing and mp and ep:
        base=Path(manifest.get("_manifest_dir",data))
        mp=Path(mp); ep=Path(ep)
        if not mp.is_absolute(): mp=base/mp
        if not ep.is_absolute(): ep=base/ep
        pairs[(str(mp),str(ep))]=(missing,outputs.get("monthly_maxima_sha256"),
                                  outputs.get("event_clusters_sha256"))
    months, events, loaded_m, loaded_e = [], [], set(), set()
    for (mp, ep), (reps,msha,esha) in pairs.items():
        mp, ep = Path(mp), Path(ep)
        if not mp.exists() or not ep.exists():
            continue
        for path,expected in ((mp,msha),(ep,esha)):
            if expected:
                key=str(path.resolve())
                actual=_HASH_CACHE.get(key)
                if actual is None:
                    actual=sha256(path); _HASH_CACHE[key]=actual
                if actual.lower()!=str(expected).lower():
                    raise ValueError(f"source history parquet SHA-256 mismatch: {path}")
        selected = sorted(reps)
        mf = pd.read_parquet(mp, filters=[("replication_id", "in", selected)])
        ef = pd.read_parquet(ep, filters=[("replication_id", "in", selected)])
        if not mf.empty:
            months.append(mf); loaded_m.update(map(int, mf.replication_id.unique()))
        if not ef.empty:
            events.append(ef); loaded_e.update(map(int, ef.replication_id.unique()))
    loaded=loaded_m & loaded_e
    if months:
        months=[f[f.replication_id.isin(loaded)] for f in months]
    if events:
        events=[f[f.replication_id.isin(loaded)] for f in events]
    return (pd.concat(months, ignore_index=True) if months else pd.DataFrame(),
            pd.concat(events, ignore_index=True) if events else pd.DataFrame(), loaded)


def frame_hash(monthly, events, rep):
    h = hashlib.sha256()
    for frame, cols in [(monthly, ["month_id", "M_fixed15_kw"]), (events, ["month_id", "heat_id", "cluster_max_kw"])]:
        part = frame[frame.replication_id.eq(rep)][cols].sort_values(cols[:2]).reset_index(drop=True)
        h.update(pd.util.hash_pandas_object(part, index=False).values.tobytes())
    return h.hexdigest()


def law_params(d):
    name = type(d).__name__
    if name == "TailProcessDistribution":
        return {"class": name, "threshold_kw": d.threshold, "xi": d.xi, "sigma_kw": d.sigma,
                "baseline_monthly_maxima_kw": d.baseline.tolist(), "count_model": d.count_model,
                "count_mean": d.count_mean, "count_variance": d.count_variance,
                "nb_size": d.nb_size, "nb_probability": d.nb_probability}
    if isinstance(d, NonnegativeGEV):
        return {"class": name, "shape_c": d.shape_c, "xi": d.xi, "loc_kw": d.loc,
                "scale_kw": d.scale, "raw_negative_mass": d.f0,
                "natural_support_upper_kw": d.upper if math.isfinite(d.upper) else None,
                "artificial_upper_truncation_kw": None,
                "max_adaptive_quad_abs_error": d.max_quad_abs_error}
    if name == "TruncatedGEV":
        return {"class": name, "shape_c": d.shape_c, "loc_kw": d.loc, "scale_kw": d.scale,
                "lower_kw": 0.0, "upper_kw": d.upper}
    if name == "NonnegativeGaussianKDE":
        return {"class": name, "sample_kw": d.sample.tolist(), "bandwidth_kw": d.bandwidth}
    if name == "DiscreteDistribution":
        return {"class": name, "support_kw": d.support.tolist(), "probabilities": d.probabilities.tolist()}
    return {"class": name}


def make_no_upper_gev(engine, maxima):
    c, loc, scale = engine._fit_gev_finite_mean(maxima)
    d = NonnegativeGEV(c, loc, scale)
    if d.f0 > .01:
        raise ValueError("gev_negative_support_gate")
    return d, {"fit_status": "PASS_LMOM_FINITE_MEAN_NO_UPPER", "xi_hat": d.xi,
               "shape_c_hat": c, "loc_hat_kw": loc, "sigma_hat_kw": scale,
               "negative_mass_probability": d.f0, "upper_truncation_probability": 0.0,
               "finite_mean_gate_pass": True, "finite_variance_flag": bool(d.xi < .5)}


def process_batch(task):
    engine = load_engine(Path(task["base"]))
    data = Path(task["data"])
    manifest = load_manifest(Path(task["run_manifest"]))
    manifest["_manifest_dir"]=str(Path(task["run_manifest"]).resolve().parent)
    reps = list(range(task["start"], task["end"]))
    all_m, all_e, loaded = load_raw(data, manifest, reps)
    settings = yaml.safe_load((Path(task["base"])/"configs/revision.yaml").read_text(encoding="utf-8"))["revision"]
    settings["seed_root"] = task["seed_root"]
    expected_months=int(settings["training_months"])
    valid_loaded=set()
    for rep in sorted(loaded):
        mrep=all_m[all_m.replication_id.eq(rep)]
        erep=all_e[all_e.replication_id.eq(rep)]
        valid=(len(mrep)==expected_months and mrep.month_id.nunique()==expected_months
               and np.isfinite(mrep.M_fixed15_kw.to_numpy(float)).all()
               and not erep.empty and erep.month_id.notna().all()
               and np.isfinite(erep.cluster_max_kw.to_numpy(float)).all()
               and erep.heat_id.notna().all()
               and not erep.duplicated(["month_id","heat_id"]).any()
               and set(erep.month_id.unique()).issubset(set(mrep.month_id.unique())))
        if valid:
            valid_loaded.add(rep)
    loaded=valid_loaded
    missing=sorted(set(reps)-loaded)
    if missing:
        raise ValueError(f"raw-history manifest is incomplete/invalid for requested replication IDs: {missing[:20]}")
    rate = float(manifest.get("tail_rate_per_month",
                 manifest.get("input_signature_payload",{}).get("tail_rate_per_month",
                 settings["dgp"]["poisson_tail_rate_per_month"])))
    rows, fit_records, inputs, histories = [], [], {}, []
    for rep in reps:
        if rep in loaded:
            m = all_m[all_m.replication_id.eq(rep)].drop(columns=["replication_id","seed_root"], errors="ignore").copy()
            e = all_e[all_e.replication_id.eq(rep)].drop(columns=["replication_id","seed_root"], errors="ignore").copy()
        else:
            raise RuntimeError(f"missing raw history {rep}; finish offline history generation before support refits")
        m = m.sort_values("month_id").reset_index(drop=True)
        e = e.sort_values(["month_id","heat_id"]).reset_index(drop=True)
        inputs[str(rep)] = frame_hash(pd.concat([m.assign(replication_id=rep)]), pd.concat([e.assign(replication_id=rep)]), rep)
        histories.append((rep,m,e))
    input_sha=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    stem=f"rep_{reps[0]:05d}_{reps[-1]:05d}"
    meta_path=Path(task["work"])/f"{stem}_meta.json"
    csv_path,fits_path=Path(task["work"])/f"{stem}.csv",Path(task["work"])/f"{stem}_fits.jsonl"
    if task.get("resume") and meta_path.exists() and csv_path.exists() and fits_path.exists():
        try:
            old=json.loads(meta_path.read_text(encoding="utf-8"))
            if (old.get("status")=="PASS" and old.get("signature")==task["signature"]
                and old.get("input_sha256")==input_sha and old.get("start")==reps[0]
                and old.get("end")==reps[-1]+1 and sha256(csv_path)==old.get("csv_sha256")
                and sha256(fits_path)==old.get("fits_sha256")):
                return {"skipped":True,"record":old}
        except Exception:
            pass
    for rep,m,e in histories:
        maxima = m.M_fixed15_kw.to_numpy(float)
        # Construct an oracle before fitting only when the legacy policy needs
        # its q_0.999-derived bound. The no-upper and training-bound fits do not
        # inspect oracle quantities; evaluation truth is built after their fits.
        oracle = (engine.OracleDistribution(engine.dgp_from_settings(settings, rate))
                  if "legacy" in task["policies"] else None)
        legacy_B = 10.0*float(oracle.ppf(.999)) if oracle is not None else None
        policies = []
        if "no_upper" in task["policies"]:
            policies.append(("no_upper", "no_upper", None))
        if "legacy" in task["policies"]:
            policies.append(("legacy_oracle_B", "legacy", float(legacy_B)))
        if "training" in task["policies"]:
            policies.extend((f"training_B_c{c:g}", "training", c*float(maxima.max())) for c in task["multipliers"])
        fitted_policies = []
        for label, kind, B in policies:
            tail, td, full = fit_tail(engine,m,e,settings,rep,kind,B)
            if kind == "no_upper":
                try:
                    gev, gd = make_no_upper_gev(engine,maxima)
                except Exception as exc:
                    gev, gd = engine.DiscreteDistribution(maxima), {"fit_status":"FALLBACK_EMPIRICAL","fallback_reason":str(exc)}
            else:
                gev, gd = engine.fit_gev_exact(maxima,float(B))
                try:
                    c,loc,scale=engine._fit_gev_finite_mean(maxima)
                    gd.update({"shape_c_hat":c,"loc_hat_kw":loc,"sigma_hat_kw":scale,"xi_hat":-c})
                except Exception:
                    pass
            event_emp, ed=engine.fit_event_empirical(e)
            kde,kd=engine.fit_kde_exact(maxima)
            emp=engine.DiscreteDistribution(maxima)
            methods=[("TAIL",tail,td,full),("EVENT-EMP",event_emp,ed,{}),("GEV",gev,gd,{}),
                     ("KDE",kde,kd,{}),("EMP",emp,{"fit_status":"PASS"},{})]
            fitted_policies.append((label,kind,B,methods))
        if oracle is None:
            oracle = engine.OracleDistribution(engine.dgp_from_settings(settings, rate))
        cells = engine.policies_and_cells(settings, oracle, alpha120_only=False)
        for label,kind,B,methods in fitted_policies:
            for method,d,diag,full_diag in methods:
                fit_records.append({"replication_id":rep,"support_policy":label,"method_id":method,
                                    "policy_kind":kind,"upper_B_kw":B,"fit_diagnostics":diag,
                                    "candidate_diagnostics":full_diag,"law":law_params(d)})
                mu=float(d.mean())
                q99=float(d.ppf(.99)); mean_true=float(oracle.mean()); q99_true=float(oracle.ppf(.99))
                for cell in cells:
                    p=cell["policy"]; truth=cell["oracle"]; decision=engine.optimized_decision(d,cell["S_kva"],p)
                    costs, tc=decision["mode_costs"],truth["mode_costs"]
                    pstar=float(p.contract_probability); q=float(d.ppf(pstar)); qt=float(oracle.ppf(pstar))
                    h_est=float(d.stop_loss(p.alpha*truth["contract_D_kw"]))
                    htrue=float(oracle.stop_loss(p.alpha*truth["contract_D_kw"]))
                    selected_true=engine.true_cost_of_estimated_action(decision["selected_mode"],decision["contract_D_kw"],oracle,cell["S_kva"],p)
                    regret=selected_true-float(truth["best_value_cny"])
                    oracle_es=qt+float(oracle.stop_loss(qt))/(1-pstar)
                    estimated_es=q+float(d.stop_loss(q))/(1-pstar)
                    err={engine.CAPACITY:costs[engine.CAPACITY]-tc[engine.CAPACITY],
                         engine.ACTUAL:costs[engine.ACTUAL]-tc[engine.ACTUAL],
                         engine.CONTRACT:costs[engine.CONTRACT]-tc[engine.CONTRACT]}
                    rows.append({"replication_id":rep,"tail_rate_per_month":rate,
                        "tail_event_probability_per_month":1-math.exp(-rate),"method_id":method,
                        "information_set":{"TAIL":"event_level","EVENT-EMP":"event_level","GEV":"monthly_maxima",
                                           "KDE":"monthly_maxima","EMP":"monthly_maxima"}[method],
                        "model_class":{"TAIL":"GPD_count_pipeline","EVENT-EMP":"nonparametric_event_maximum",
                                       "GEV":"GEV","KDE":"Gaussian_KDE","EMP":"empirical"}[method],
                        "support_policy":label,"fit_status":diag.get("fit_status","PASS"),
                        "fallback_reason":diag.get("fallback_reason"),"policy_id":p.policy_id,"alpha":p.alpha,
                        "kappa":p.kappa,"cell_id":cell["cell_id"],"utilization":cell["utilization"],
                        "S_kva":cell["S_kva"],"oracle_mode":truth["selected_mode"],"selected_mode":decision["selected_mode"],
                        "action_correct":decision["selected_mode"]==truth["selected_mode"],
                        "oracle_margin_cny":truth["margin_cny"],"oracle_best_value_cny":truth["best_value_cny"],
                        "estimated_best_value_cny":decision["best_value_cny"],
                        "maximum_mode_value_error_cny":max(abs(v) for v in err.values()),
                        "d_mv":2*max(abs(v) for v in err.values())/truth["margin_cny"],
                        "d_mv_below_one":2*max(abs(v) for v in err.values())/truth["margin_cny"]<1,
                        "true_mean_kw":mean_true,"estimated_mean_kw":mu,"mean_relative_error":(mu-mean_true)/mean_true,
                        "true_q99_kw":q99_true,"estimated_q99_kw":q99,"q99_relative_error":(q99-q99_true)/q99_true,
                        "estimated_contract_D_kw":decision["contract_D_kw"],"oracle_contract_D_kw":truth["contract_D_kw"],
                        "contract_D_error_over_S":(decision["contract_D_kw"]-truth["contract_D_kw"])/cell["S_kva"],
                        "stoploss_threshold_kw":p.alpha*truth["contract_D_kw"],
                        "true_stoploss_kw":htrue,"estimated_stoploss_kw":h_est,"stoploss_relative_error":(h_est-htrue)/htrue,
                        "contract_quantile_probability_p_star":pstar,
                        "oracle_contract_quantile_p_star_kw":qt,"estimated_contract_quantile_p_star_kw":q,
                        "contract_quantile_error_p_star_kw":q-qt,
                        "contract_quantile_relative_error_p_star":(q-qt)/qt,
                        "oracle_es_p_star_kw":oracle_es,"estimated_es_p_star_kw":estimated_es,
                        "es_relative_error_p_star":(estimated_es-oracle_es)/oracle_es,
                        "estimated_mode_cost_capacity":costs[engine.CAPACITY],
                        "estimated_mode_cost_actual":costs[engine.ACTUAL],
                        "estimated_mode_cost_contract":costs[engine.CONTRACT],
                        "delta_capacity_cny":err[engine.CAPACITY],"delta_actual_cny":err[engine.ACTUAL],
                        "delta_contract_cny":err[engine.CONTRACT],
                        "regret_cny_per_month":max(0.0,regret),
                        "regret_over_oracle_value":max(0.0,regret)/truth["best_value_cny"],
                        "regret_over_capacity_fee":max(0.0,regret)/(p.capacity_rate*cell["S_kva"]),
                        "draw_count_for_final_functionals":0,"final_functional_evaluation":"deterministic",
                        "selected_threshold_quantile":diag.get("selected_threshold_quantile",np.nan),
                        "threshold_kw":diag.get("threshold_kw",np.nan),"xi_hat":diag.get("xi_hat",np.nan),
                        "sigma_hat_kw":diag.get("sigma_hat_kw",np.nan),"count_model":diag.get("count_model","not_applicable"),
                        "refit_used_all_months":diag.get("refit_used_all_months",False),
                        "finite_mean_gate_pass":diag.get("finite_mean_gate_pass",True),
                        "finite_variance_flag":diag.get("finite_variance_flag",True)})
                fit_records[-1]["law"] = law_params(d)
    return {"skipped":False,"start":reps[0],"end":reps[-1]+1,"rows":rows,"fits":fit_records,"input_hashes":inputs,
            "loaded":sorted(loaded),"regenerated":sorted(set(reps)-loaded)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-repo",type=Path,default=DEFAULT_BASE)
    ap.add_argument("--data",type=Path,default=DEFAULT_DATA)
    ap.add_argument("--output",type=Path,default=PACKAGE/"data"/"primary_no_upper_20261001")
    ap.add_argument("--run-manifest",type=Path)
    ap.add_argument("--seed-root",type=int,default=20261001)
    ap.add_argument("--start",type=int,required=False)
    ap.add_argument("--end",type=int,required=False)
    ap.add_argument("--chunk-size",type=int,default=5)
    ap.add_argument("--workers",type=int,default=1)
    ap.add_argument("--policies",default="no_upper")
    ap.add_argument("--training-multipliers",type=lambda s:[float(x) for x in s.split(",")],default=[10.,20.,50.])
    ap.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--self-check",action="store_true")
    a=ap.parse_args()
    if a.self_check:
        print(json.dumps(self_check(),indent=2)); return
    if a.start is None or a.end is None or a.start<0 or a.end<=a.start:
        ap.error("provide a nonempty [--start,--end) range")
    if not 1<=a.workers<=15 or a.chunk_size<1: ap.error("workers must be 1..15 and chunk-size positive")
    policies=[x.strip() for x in a.policies.split(",") if x.strip()]
    if policies != ["no_upper"]:
        ap.error("the CV-CRPS primary runner supports only the no_upper policy")
    if any(not math.isfinite(x) or x<=0 for x in a.training_multipliers):
        ap.error("training multipliers must be positive finite values")
    data=a.data.resolve()
    run_path=a.run_manifest.resolve() if a.run_manifest else data/"run_manifest.json"
    manifest=json.loads(run_path.read_text(encoding="utf-8"))
    input_payload=manifest.get("input_signature_payload",{})
    source_seed=manifest.get("seed_root",input_payload.get("seed_root",a.seed_root))
    if int(source_seed)!=a.seed_root:
        ap.error("seed-root differs from source run manifest")
    cfg=yaml.safe_load((a.base_repo/"configs/revision.yaml").read_text(encoding="utf-8"))["revision"]
    if input_payload:
        payload_signature=hashlib.sha256(json.dumps(
            input_payload,sort_keys=True,separators=(",",":"),ensure_ascii=False
        ).encode("utf-8")).hexdigest()
        if not manifest.get("input_signature") or manifest["input_signature"].lower()!=payload_signature:
            ap.error("offline history input_signature does not match its canonical payload")
        if int(input_payload.get("training_months",cfg["training_months"]))!=int(cfg["training_months"]):
            ap.error("source history training-month count differs from base revision.yaml")
        if input_payload.get("dgp_settings") and safe(input_payload["dgp_settings"])!=safe(cfg["dgp"]):
            ap.error("source history DGP settings differ from base revision.yaml")
        configured_rate=float(cfg["dgp"]["poisson_tail_rate_per_month"])
        declared_rate=float(input_payload.get("tail_rate_per_month",configured_rate))
        manifest_rate=float(manifest.get("tail_rate_per_month",declared_rate))
        if not math.isclose(declared_rate,configured_rate,rel_tol=0,abs_tol=1e-12) or not math.isclose(manifest_rate,configured_rate,rel_tol=0,abs_tol=1e-12):
            ap.error("source history tail rate differs from base revision.yaml")
        if input_payload.get("rng_namespace") not in (None,[2,0,"replication_id",10]):
            ap.error("source history RNG namespace differs from the registered primary namespace")
    declared_revision_sha=input_payload.get("revision_yaml_sha256")
    actual_revision_sha=sha256(a.base_repo/"configs/revision.yaml")
    if declared_revision_sha and declared_revision_sha.lower()!=actual_revision_sha.lower():
        ap.error("base-repo revision.yaml does not match the source history manifest")
    declared_config_sha=manifest.get("config_sha256")
    if declared_revision_sha and declared_config_sha and declared_config_sha.lower()!=actual_revision_sha.lower():
        ap.error("manifest config_sha256 does not match base revision.yaml")
    if declared_config_sha and not declared_revision_sha:
        primary_cfg=PACKAGE/"config"/"primary_10000.yaml"
        if declared_config_sha.lower()!=actual_revision_sha.lower() and (
                not primary_cfg.exists() or declared_config_sha.lower()!=sha256(primary_cfg).lower()):
            ap.error("source config hash matches neither base revision.yaml nor the extension run config")
    source_files=input_payload.get("base_src_extreme_demand_files",[])
    for record in source_files:
        source_file=a.base_repo/record["path"]
        if not source_file.is_file() or sha256(source_file).lower()!=record["sha256"].lower():
            ap.error(f"base source file differs from offline history input signature: {record['path']}")
    cfg_sig=hashlib.sha256(json.dumps(safe(cfg),sort_keys=True).encode()).hexdigest()
    sig_obj={"source_signature":manifest.get("signature_sha256"),"source_config_sha256":manifest.get("config_sha256"),
        "source_input_signature":manifest.get("input_signature"),
        "source_manifest_sha256":sha256(run_path),"seed_root":a.seed_root,"base_engine_sha256":sha256(a.base_repo/"scripts/run_revision_experiment.py"),
        "revision_config_sha256":cfg_sig,"policies":policies,"multipliers":a.training_multipliers,
        "script_sha256":sha256(Path(__file__).resolve())}
    signature=hashlib.sha256(json.dumps(sig_obj,sort_keys=True).encode()).hexdigest()
    out=a.output.resolve(); work=out/"support_policy_audit"/signature[:16]; work.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for lo in range(a.start,a.end,a.chunk_size):
        hi=min(a.end,lo+a.chunk_size)
        tasks.append({"base":str(a.base_repo.resolve()),"data":str(data),"run_manifest":str(run_path),
            "work":str(work),"signature":signature,"resume":a.resume,
            "start":lo,"end":hi,"seed_root":a.seed_root,"policies":policies,"multipliers":a.training_multipliers})
    if "no_upper" in policies and (
        str(manifest.get("status","")).upper()!="PASS"
        or not input_payload.get("revision_yaml_sha256")
        or not manifest.get("input_signature")
    ):
        ap.error("no-upper primary refit requires a complete PASS offline-history manifest with a revision config hash")
    source_total=int(manifest.get("replications",input_payload.get("replications",a.end)))
    if a.end>source_total:
        ap.error(f"requested end {a.end} exceeds source histories {source_total}")
    audit={"status":"RUNNING","signature":signature,"source_run_signature":manifest.get("signature_sha256") or manifest.get("input_signature"),
        "source_manifest":str(run_path),"source_manifest_sha256":sha256(run_path),"source_config_sha256":manifest.get("config_sha256"),
        "seed_root":a.seed_root,"range":[a.start,a.end],"chunk_size":a.chunk_size,"workers":a.workers,
        "policies":policies,"training_multipliers":a.training_multipliers,"chunks":[]}
    audit_path=work/"audit_manifest.json"
    atomic_text(audit_path,json.dumps(audit,indent=2))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(process_batch,t):t for t in tasks}
        for future in as_completed(futures):
            result=future.result()
            if result.get("skipped"):
                audit["chunks"].append(result["record"]); audit["chunks"].sort(key=lambda x:x["start"])
                atomic_text(audit_path,json.dumps(safe(audit),indent=2))
                print(f"[SKIP verified] {result['record']['start']}:{result['record']['end']}",flush=True)
                continue
            lo,hi=result["start"],result["end"]
            stem=f"rep_{lo:05d}_{hi-1:05d}"; csvp=work/f"{stem}.csv"; fitp=work/f"{stem}_fits.jsonl"
            atomic_csv(result["rows"],csvp)
            atomic_text(fitp,"".join(json.dumps(safe(x),ensure_ascii=False,allow_nan=False)+"\n" for x in result["fits"]))
            # Input hash is computed from the actual selected raw rows, so resume
            # checks both policy/config signature and source history contents.
            record={"status":"PASS","signature":signature,"start":lo,"end":hi,
                "input_sha256":hashlib.sha256(json.dumps(result["input_hashes"],sort_keys=True).encode()).hexdigest(),
                "input_hashes":result["input_hashes"],"loaded_raw_replications":result["loaded"],
                "regenerated_replications":result["regenerated"],"decision_rows":len(result["rows"]),
                "fit_records":len(result["fits"]),"csv":csvp.name,"fits":fitp.name,
                "csv_sha256":sha256(csvp),"fits_sha256":sha256(fitp)}
            atomic_text(work/f"{stem}_meta.json",json.dumps(record,indent=2))
            audit["chunks"].append(record); audit["chunks"].sort(key=lambda x:x["start"])
            atomic_text(audit_path,json.dumps(safe(audit),indent=2))
            print(f"[PASS] {lo}:{hi}",flush=True)
    # Export analyzer-compatible raw decisions once a complete no-upper run exists.
    total_reps=manifest.get("replications") or input_payload.get("replications")
    if policies==["no_upper"] and a.start==0 and total_reps and a.end==int(total_reps)==10000:
        raw=out/"raw"/"primary_decisions_10000.csv.gz"; raw.parent.mkdir(parents=True,exist_ok=True)
        tmp=raw.with_suffix(raw.suffix+".tmp")
        with tmp.open("wb") as f:
            with gzip.GzipFile(fileobj=f,mode="wb",compresslevel=6,mtime=0) as gz:
                wrote=False
                for lo in range(a.start,a.end,a.chunk_size):
                    hi=min(a.end,lo+a.chunk_size); stem=f"rep_{lo:05d}_{hi-1:05d}"
                    p=work/f"{stem}.csv"; meta=work/f"{stem}_meta.json"
                    if not p.exists() or not meta.exists() or json.loads(meta.read_text(encoding="utf-8")).get("status")!="PASS":
                        raise RuntimeError(f"incomplete no-upper history chunk {lo}:{hi}")
                    frame=pd.read_csv(p)
                    expected_reps=set(range(lo,hi))
                    expected_methods={"TAIL","EVENT-EMP","GEV","KDE","EMP"}
                    if len(frame)!=(hi-lo)*45:
                        raise RuntimeError(f"wrong row count for chunk {lo}:{hi}: {len(frame)}")
                    if frame[["replication_id","method_id","cell_id"]].duplicated().any():
                        raise RuntimeError(f"duplicate decision keys for chunk {lo}:{hi}")
                    if set(frame.replication_id.unique())!=expected_reps:
                        raise RuntimeError(f"replication IDs incomplete in chunk {lo}:{hi}")
                    if set(frame.method_id.unique())!=expected_methods or frame.cell_id.nunique()!=9:
                        raise RuntimeError(f"method × cell domain is incomplete in chunk {lo}:{hi}")
                    grid=frame.groupby(["method_id","cell_id"],observed=True).size()
                    if len(grid)!=45 or not (grid==(hi-lo)).all():
                        raise RuntimeError(f"5×9 decision grid is incomplete in chunk {lo}:{hi}")
                    if set(frame.support_policy.unique())!={"no_upper"}:
                        raise RuntimeError(f"unexpected support policy in chunk {lo}:{hi}")
                    gz.write(frame.to_csv(index=False,header=not wrote).encode("utf-8")); wrote=True
        os.replace(tmp,raw)
        audit["merged_decisions"]=str(raw); audit["merged_decisions_sha256"]=sha256(raw)
    audit["status"]="PASS"
    atomic_text(audit_path,json.dumps(safe(audit),indent=2))
    print(json.dumps({"status":"PASS","output":str(work),"chunks_processed":len(tasks)},indent=2))


if __name__=="__main__":
    main()
