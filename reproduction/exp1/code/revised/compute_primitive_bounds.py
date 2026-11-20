#!/usr/bin/env python3
"""Compute distribution distances and primitive value-certificate inputs.

This is an independent post-processing stage. It uses only the corrected
10,000-history primary run, reconstructs each fitted law, and writes into a
new result directory. It never edits the manuscript or source run outputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

# The surrounding experiment plan caps process workers explicitly. Prevent
# BLAS/OpenMP from multiplying each process into additional threads.
for _thread_env in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_env]="1"

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import special, stats


_ENGINE = None
_AUDIT = None
_SETTINGS = None
_ORACLE = None
_GL_X = None
_GL_W = None
_H_GL_X = None
_H_GL_W = None
_W1_ORDER = None
_H_GRID = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    os.replace(tmp,path)


def load_engine(base_repo: Path):
    sys.path.insert(0, str(base_repo / "src"))
    p = base_repo / "scripts" / "run_revision_experiment.py"
    spec = importlib.util.spec_from_file_location("primitive_bounds_engine", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load primary engine: {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def init_worker(base_repo: str, audit_script: str, w1_order: int, h_grid: int, h_quad: int):
    global _ENGINE, _AUDIT, _SETTINGS, _ORACLE
    global _GL_X, _GL_W, _H_GL_X, _H_GL_W, _W1_ORDER, _H_GRID
    _ENGINE = load_engine(Path(base_repo))
    spec = importlib.util.spec_from_file_location("primitive_bounds_audit", audit_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load no-upper law class module: {audit_script}")
    _AUDIT = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = _AUDIT
    spec.loader.exec_module(_AUDIT)
    _SETTINGS = _ENGINE.load_settings()
    dgp = _ENGINE.dgp_from_settings(_SETTINGS)
    _ORACLE = _ENGINE.OracleDistribution(dgp)
    x, w = np.polynomial.legendre.leggauss(int(w1_order))
    _GL_X, _GL_W = 0.5 * (x + 1.0), 0.5 * w
    xh, wh = np.polynomial.legendre.leggauss(int(h_quad))
    _H_GL_X, _H_GL_W = 0.5 * (xh + 1.0), 0.5 * wh
    _W1_ORDER, _H_GRID = int(w1_order), int(h_grid)


def true_cdf_vec(x: np.ndarray) -> np.ndarray:
    d = _ORACLE.dgp
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    below = (x >= d.baseline_lower_kw) & (x < d.threshold_kw)
    if np.any(below):
        z = (x[below] - d.baseline_lower_kw) / (d.threshold_kw - d.baseline_lower_kw)
        out[below] = math.exp(-d.tail_rate) * stats.beta.cdf(z, d.beta_a, d.beta_b) ** d.baseline_events
    upper = x >= d.threshold_kw
    if np.any(upper):
        s = stats.genpareto.sf(x[upper] - d.threshold_kw, d.xi, loc=0.0, scale=d.sigma_kw)
        out[upper] = np.exp(-d.tail_rate * s)
    return np.clip(out, 0.0, 1.0)


def true_ppf_vec(p: np.ndarray) -> np.ndarray:
    d = _ORACLE.dgp
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    out = np.full_like(p, d.baseline_lower_kw)
    p0 = math.exp(-d.tail_rate)
    low = (p > 0.0) & (p < p0)
    if np.any(low):
        bp = (p[low] / p0) ** (1.0 / d.baseline_events)
        out[low] = d.baseline_lower_kw + (d.threshold_kw - d.baseline_lower_kw) * stats.beta.ppf(bp, d.beta_a, d.beta_b)
    high = p >= p0
    if np.any(high):
        ph = np.minimum(p[high], np.nextafter(1.0, 0.0))
        surv = -np.log(ph) / d.tail_rate
        out[high] = d.threshold_kw + stats.genpareto.isf(surv, d.xi, loc=0.0, scale=d.sigma_kw)
    return out


def true_ppf_from_survival(survival: np.ndarray) -> np.ndarray:
    """Stable oracle upper-tail quantiles without forming 1-survival.

    For p close to one, `1 - survival` rounds to exactly one in float64 and a
    conventional PPF returns infinity.  The true DGP has an exact GPD max-law
    representation above its threshold, so invert its survival probability
    directly with `genpareto.isf`.
    """
    s=np.asarray(survival,dtype=float)
    d=_ORACLE.dgp
    if np.any((s<=0.0)|(s>=1.0)):
        raise ValueError("upper-tail survival probabilities must lie strictly inside (0,1)")
    gpd_survival=-np.log1p(-s)/float(d.tail_rate)
    return float(d.threshold_kw)+stats.genpareto.isf(gpd_survival,float(d.xi),loc=0.0,scale=float(d.sigma_kw))


def estimated_ppf_from_survival(engine,dist,survival:np.ndarray) -> np.ndarray:
    """Stable upper-tail quantiles for fitted GEV and tail-process laws."""
    s=np.asarray(survival,dtype=float)
    if np.any((s<=0.0)|(s>=1.0)):
        raise ValueError("upper-tail survival probabilities must lie strictly inside (0,1)")
    if isinstance(dist,_AUDIT.NonnegativeGEV):
        return np.maximum(0.0,stats.genextreme.isf(s,dist.shape_c,loc=dist.loc,scale=dist.scale))
    if isinstance(dist,engine.TailProcessDistribution):
        if dist.count_model=="poisson":
            gpd_survival=-np.log1p(-s)/float(dist.count_mean)
        else:
            odds=(1.0-float(dist.nb_probability))/float(dist.nb_probability)
            gpd_survival=np.expm1(-np.log1p(-s)/float(dist.nb_size))/odds
        return float(dist.threshold)+stats.genpareto.isf(gpd_survival,float(dist.xi),loc=0.0,scale=float(dist.sigma))
    if isinstance(dist,engine.NonnegativeGaussianKDE):
        # This family has a Gaussian tail. Bisection against a p rounded to
        # one can lose the last few ulps, so retain a survival-aware target.
        # The existing finite bracket is a conservative numerical endpoint.
        p=1.0-s
        p=np.minimum(p,np.nextafter(1.0,0.0))
        return estimated_ppf_vec(engine,dist,p)
    raise TypeError(type(dist))


def max_process_stoploss_vec(t: np.ndarray, threshold: float, xi: float, sigma: float,
                             count_model: str, count_mean: float,
                             nb_size: float | None, nb_probability: float | None,
                             baseline: np.ndarray | None = None) -> np.ndarray:
    """Vectorized exact stop-loss formula for maxima of fitted marked counts.

    The remaining one-dimensional integral uses Gauss-Legendre quadrature on
    [0,1] after the same endpoint-regularizing transform used by the engine.
    """
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    above = t >= threshold
    if np.any(above):
        s = stats.genpareto.sf(t[above] - threshold, xi, loc=0.0, scale=sigma)
        out[above] = _count_tail_integral_vec(s, xi, sigma, count_model, count_mean,
                                              nb_size, nb_probability)
    below = ~above
    if np.any(below):
        if baseline is None:
            raise ValueError("baseline maxima required below the fitted threshold")
        if count_model == "poisson":
            pzero = math.exp(-count_mean)
        else:
            odds = (1.0 - float(nb_probability)) / float(nb_probability)
            pzero = (1.0 + odds) ** (-float(nb_size))
        tb = t[below]
        cdf_area = np.maximum(threshold - np.maximum(baseline[None, :], tb[:, None]), 0.0).mean(axis=1)
        tail_all = _count_tail_integral_vec(np.ones(1), xi, sigma, count_model,
                                             count_mean, nb_size, nb_probability)[0]
        out[below] = (threshold - tb) - pzero * cdf_area + tail_all
    return out


def _count_tail_integral_vec(s: np.ndarray, xi: float, sigma: float, count_model: str,
                             count_mean: float, nb_size: float | None,
                             nb_probability: float | None) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)
    positive = s > 0.0
    if not positive.any() or count_mean <= 0.0:
        return out
    sp = s[positive]
    power = 1.0 / (1.0 - float(xi))
    u = sp[:, None] * (_H_GL_X[None, :] ** power)
    if count_model == "poisson":
        exceed = -np.expm1(-float(count_mean) * u)
    else:
        odds = (1.0 - float(nb_probability)) / float(nb_probability)
        exceed = -np.expm1(-float(nb_size) * np.log1p(odds * u))
    ratio = np.divide(exceed, u, out=np.full_like(exceed, float(count_mean)), where=u > 1e-280)
    out[positive] = sigma * (sp ** (1.0 - xi)) * power * (ratio @ _H_GL_W)
    return out


def true_stoploss_vec(t: np.ndarray) -> np.ndarray:
    d = _ORACLE.dgp
    t = np.asarray(t, dtype=float)
    above = t >= d.threshold_kw
    out = np.empty_like(t)
    if np.any(above):
        s = stats.genpareto.sf(t[above] - d.threshold_kw, d.xi, loc=0.0, scale=d.sigma_kw)
        out[above] = _count_tail_integral_vec(s, d.xi, d.sigma_kw, "poisson",
                                              d.tail_rate, None, None)
    if np.any(~above):
        # Below the generating threshold F(x) <= exp(-rate), hence
        # 0 <= H(t) - (E[X]-t) <= t*exp(-rate). The omitted term is bounded.
        out[~above] = _ORACLE.mean() - t[~above]
    return out


def gev_stoploss_vec(dist, t: np.ndarray) -> np.ndarray:
    """Stop-loss for stored NonnegativeGEV laws, conditioned only at zero."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    below0 = t < 0
    out[below0] = dist.mean() - t[below0]
    work = np.maximum(t, 0.0)
    xi, loc, scale = dist.xi, dist.loc, dist.scale
    below_lower = work <= dist.lower
    out[below_lower & ~below0] = np.maximum(0.0, dist.raw_mean-work[below_lower & ~below0]) / dist.a
    above_upper = work >= dist.upper
    interior = ~(below_lower | above_upper | below0)
    if not np.any(interior):
        return np.maximum(out, 0.0)
    x = work[interior]
    if abs(xi) < 1e-5:
        # This near-Gumbel case is uncommon; retain the source class's stable
        # scalar integration exactly where its own implementation requires it.
        out[interior] = np.fromiter((dist.stop_loss(float(v)) for v in x), float, count=x.size)
        return np.maximum(out, 0.0)
    if xi == 0.0:
        v = np.exp(-(x-loc)/scale)
        raw = scale * (-np.log(np.maximum(v, np.finfo(float).tiny)) + np.euler_gamma
                       + special.exp1(np.maximum(v, np.finfo(float).tiny)))
    else:
        logbase = np.log1p(xi*(x-loc)/scale)
        logv = -logbase/xi
        huge = logv > 700
        v = np.exp(np.clip(logv, -745, 700))
        A = -np.expm1(-v)
        shape = 1.0-xi
        bracket = special.gamma(shape)*special.gammainc(shape, v)-A
        large = v > 50
        if np.any(large):
            bracket[large] = math.expm1(float(special.gammaln(shape)))
        raw = (loc-x)*A + scale/xi*bracket
        raw[huge] = (np.maximum(0.0, dist.raw_mean-x[huge]) if xi > 0 else 0.0)
    out[interior] = np.maximum(raw, 0.0)/dist.a
    return np.maximum(out, 0.0)


def distribution_cdf_vec(engine, dist, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if isinstance(dist, engine.DiscreteDistribution):
        idx = np.searchsorted(dist.support, x, side="right") - 1
        out = np.zeros_like(x)
        good = idx >= 0
        out[good] = dist.cumulative[idx[good]]
        return out
    if isinstance(dist, engine.NonnegativeGaussianKDE):
        upper = special.ndtr((x[:, None] - dist.sample[None, :]) / dist.bandwidth)
        lower = special.ndtr(-dist.sample / dist.bandwidth)
        return np.clip((upper - lower[None, :]).sum(axis=1) / dist._normalizer, 0.0, 1.0)
    if isinstance(dist, _AUDIT.NonnegativeGEV):
        out = np.zeros_like(x)
        positive = x > 0.0
        if np.any(positive):
            raw = stats.genextreme.cdf(x[positive], dist.shape_c, loc=dist.loc, scale=dist.scale)
            out[positive] = np.clip((raw-dist.f0)/dist.a, 0.0, 1.0)
        return out
    if isinstance(dist, engine.TailProcessDistribution):
        baseline_cdf = (dist.baseline[None, :] <= x[:, None]).mean(axis=1)
        if dist.count_model == "poisson":
            pgf0 = math.exp(-dist.count_mean)
            pgf = lambda g: np.exp(-dist.count_mean * (1.0 - g))
        else:
            odds = (1.0 - dist.nb_probability) / dist.nb_probability
            pgf0 = math.exp(-dist.nb_size * math.log1p(odds))
            pgf = lambda g: np.exp(-dist.nb_size * np.log1p(odds * (1.0 - g)))
        out = np.empty_like(x)
        below = x < dist.threshold
        out[below] = baseline_cdf[below] * pgf0
        if np.any(~below):
            gc = stats.genpareto.cdf(x[~below] - dist.threshold, dist.xi, loc=0.0, scale=dist.sigma)
            out[~below] = baseline_cdf[~below] * pgf(gc)
        return np.clip(out, 0.0, 1.0)
    raise TypeError(type(dist))


def distribution_cdf_left(engine,dist,x:float) -> float:
    """Left limit F(x-) for laws with atoms; continuous laws use F(x)."""
    x=float(x)
    if isinstance(dist,engine.DiscreteDistribution):
        idx=np.searchsorted(dist.support,x,side="left")-1
        return 0.0 if idx<0 else float(dist.cumulative[idx])
    if isinstance(dist,engine.TailProcessDistribution):
        baseline=float(np.mean(dist.baseline<x))
        if dist.count_model=="poisson":
            pzero=math.exp(-dist.count_mean)
            pgf=lambda g:math.exp(-dist.count_mean*(1-g))
        else:
            odds=(1-dist.nb_probability)/dist.nb_probability
            pzero=(1+odds)**(-dist.nb_size)
            pgf=lambda g:(1+odds*(1-g))**(-dist.nb_size)
        if x<dist.threshold: return baseline*pzero
        gc=float(stats.genpareto.cdf(x-dist.threshold,dist.xi,loc=0,scale=dist.sigma))
        return baseline*pgf(gc)
    return float(distribution_cdf_vec(engine,dist,np.array([x]))[0])


def distribution_cdf_left_vec(engine,dist,x:np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=float)
    if isinstance(dist,engine.DiscreteDistribution):
        idx=np.searchsorted(dist.support,x,side="left")-1
        out=np.zeros_like(x); good=idx>=0; out[good]=dist.cumulative[idx[good]]
        return out
    if isinstance(dist,engine.TailProcessDistribution):
        baseline=np.searchsorted(np.sort(dist.baseline),x,side="left")/dist.baseline.size
        if dist.count_model=="poisson":
            pzero=math.exp(-dist.count_mean); pgf=lambda g:np.exp(-dist.count_mean*(1-g))
        else:
            odds=(1-dist.nb_probability)/dist.nb_probability
            pzero=(1+odds)**(-dist.nb_size); pgf=lambda g:(1+odds*(1-g))**(-dist.nb_size)
        out=np.empty_like(x); below=x<dist.threshold; out[below]=baseline[below]*pzero
        if np.any(~below):
            gc=stats.genpareto.cdf(x[~below]-dist.threshold,dist.xi,loc=0,scale=dist.sigma)
            out[~below]=baseline[~below]*pgf(gc)
        return np.clip(out,0,1)
    return distribution_cdf_vec(engine,dist,x)


def estimated_tail_stoploss_upper(engine,dist,t:float) -> float:
    """Analytic/union tail upper Hhat(t), for t beyond every finite bulk atom."""
    t=float(t)
    if isinstance(dist,engine.DiscreteDistribution):
        return float(np.dot(dist.probabilities,np.maximum(dist.support-t,0.0)))
    if isinstance(dist,engine.NonnegativeGaussianKDE):
        return float(distribution_stoploss_vec(engine,dist,np.array([t]))[0])
    if isinstance(dist,engine.TailProcessDistribution):
        if t<max(dist.threshold,float(np.max(dist.baseline))): return math.inf
        if dist.xi>0:
            s=float(stats.genpareto.sf(t-dist.threshold,dist.xi,loc=0,scale=dist.sigma))
            return dist.count_mean*dist.sigma/(1-dist.xi)*s**(1-dist.xi)
        if dist.xi==0:
            s=math.exp(-(t-dist.threshold)/dist.sigma)
        else:
            s=max(0.0,1+dist.xi*(t-dist.threshold)/dist.sigma)**(-1/dist.xi) if t<dist.threshold-dist.sigma/dist.xi else 0.0
        return dist.count_mean*dist.sigma/(1-dist.xi)*s**(1-dist.xi)
    if isinstance(dist,_AUDIT.NonnegativeGEV):
        if t<0: return math.inf
        if t>=dist.upper: return 0.0
        if dist.xi==0:
            v=math.exp(-(t-dist.loc)/dist.scale)
        else:
            base=1+dist.xi*(t-dist.loc)/dist.scale
            if base<=0: return 0.0 if dist.xi<0 else math.inf
            v=base**(-1/dist.xi)
        return dist.scale/(1-dist.xi)/dist.a*v**(1-dist.xi)
    raise TypeError(type(dist))


def true_tail_stoploss_upper(t:float) -> float:
    d=_ORACLE.dgp; t=float(t)
    if t<d.threshold_kw: return math.inf
    s=float(stats.genpareto.sf(t-d.threshold_kw,d.xi,loc=0,scale=d.sigma_kw))
    return d.tail_rate*d.sigma_kw/(1-d.xi)*s**(1-d.xi)


def cdf_w1_riemann_bounds(engine,dist,n_grid:int=512,tail_goal:float=10.0) -> dict[str,float]:
    """CDF-integral W1 interval plus an analytic union-bound tail remainder."""
    lower=max(0.0,float(_ORACLE.dgp.threshold_kw))
    if isinstance(dist,engine.TailProcessDistribution):
        lower=max(lower,dist.threshold,float(np.max(dist.baseline)))
    elif isinstance(dist,engine.DiscreteDistribution):
        lower=max(lower,float(np.max(dist.support)))
    def tail_sum(z): return estimated_tail_stoploss_upper(engine,dist,z)+true_tail_stoploss_upper(z)
    hi=max(lower+1000.0,2.0*max(1.0,lower))
    for _ in range(100):
        if tail_sum(hi)<=tail_goal: break
        hi=max(2.0*hi,hi+1000.0)
    else: raise RuntimeError(f"could not bound W1 tail for {type(dist).__name__}")
    lo=lower
    for _ in range(48):
        mid=0.5*(lo+hi)
        if tail_sum(mid)<=tail_goal: hi=mid
        else: lo=mid
    T=hi; tail=float(tail_sum(T))
    points=np.linspace(0.0,T,int(n_grid)+1)
    if isinstance(dist,engine.DiscreteDistribution):
        points=np.r_[points,dist.support[(dist.support>0)&(dist.support<T)]]
    elif isinstance(dist,engine.TailProcessDistribution):
        points=np.r_[points,dist.baseline[(dist.baseline>0)&(dist.baseline<T)],dist.threshold]
    elif isinstance(dist,_AUDIT.NonnegativeGEV) and math.isfinite(dist.upper):
        points=np.r_[points,dist.upper]
    points=np.unique(np.clip(points,0,T)); points.sort()
    a,b=points[:-1],points[1:]; dx=b-a
    fa=distribution_cdf_vec(engine,dist,a)
    fb=distribution_cdf_left_vec(engine,dist,b)
    ta=true_cdf_vec(a); tb=true_cdf_vec(b)
    g_lo=fa-tb; g_hi=fb-ta
    cell_lower=np.maximum(0.0,np.maximum(g_lo,-g_hi))
    cell_upper=np.maximum(np.abs(g_lo),np.abs(g_hi))
    lower_est=float(np.dot(dx,cell_lower))
    upper_est=float(np.dot(dx,cell_upper)+tail)
    # Covers ordinary floating summation/CDF rounding scale, not interval arithmetic.
    fp_allow=2048*np.finfo(float).eps*max(1.0,T)*(1+len(points))
    return {"lower":max(0.0,lower_est-fp_allow),"upper":upper_est+fp_allow,
            "mesh_kw":float(np.max(dx)) if len(dx) else 0.0,"tail_kw":tail,"T_kw":T,
            "fp_allowance_kw":float(fp_allow)}


def distribution_stoploss_vec(engine, dist, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if isinstance(dist, engine.DiscreteDistribution):
        idx = np.searchsorted(dist.support, t, side="right")
        prefix_p = np.r_[0.0, np.cumsum(dist.probabilities)]
        prefix_m = np.r_[0.0, np.cumsum(dist.support * dist.probabilities)]
        tailp = 1.0 - prefix_p[idx]
        tailm = float(prefix_m[-1]) - prefix_m[idx]
        return np.maximum(0.0, tailm - t * tailp)
    if isinstance(dist, engine.NonnegativeGaussianKDE):
        z = (dist.sample[None, :] - t[:, None]) / dist.bandwidth
        numer = (dist.sample[None, :] - t[:, None]) * special.ndtr(z) + dist.bandwidth * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return numer.sum(axis=1) / dist._normalizer
    if isinstance(dist, _AUDIT.NonnegativeGEV):
        return gev_stoploss_vec(dist, t)
    if isinstance(dist, engine.TailProcessDistribution):
        return max_process_stoploss_vec(t, dist.threshold, dist.xi, dist.sigma,
                                       dist.count_model, dist.count_mean,
                                       dist.nb_size, dist.nb_probability,
                                       dist.baseline)
    raise TypeError(type(dist))


def true_integrated_upper_quantile(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = true_ppf_vec(p)
    h = true_stoploss_vec(q)
    k = (1.0 - p) * q + h
    k[p >= 1.0] = 0.0
    k[p <= 0.0] = _ORACLE.mean()
    return k


def discrete_w1(engine, dist) -> tuple[float, float]:
    x = np.asarray(dist.support, dtype=float)
    p = np.asarray(dist.probabilities, dtype=float)
    edges = np.r_[0.0, np.cumsum(p)]
    edges[-1] = 1.0
    ftrue = true_cdf_vec(x)
    cross = np.clip(ftrue, edges[:-1], edges[1:])
    klo = true_integrated_upper_quantile(edges[:-1])
    khi = true_integrated_upper_quantile(edges[1:])
    hcross = true_stoploss_vec(x)
    kcross = (1.0 - cross) * x + hcross
    kcross = np.where(ftrue <= edges[:-1], klo, np.where(ftrue >= edges[1:], khi, kcross))
    lower = x * (cross - edges[:-1]) - (klo - kcross)
    upper = (kcross - khi) - x * (edges[1:] - cross)
    value = float(np.maximum(lower, 0.0).sum() + np.maximum(upper, 0.0).sum())
    # H below the DGP tail threshold uses E[X]-t and omits a nonnegative
    # correction bounded by t*exp(-lambda); carry that bound into W1.
    below_mass = float(np.maximum(0.0, np.minimum(edges[1:], math.exp(-_ORACLE.dgp.tail_rate)) - edges[:-1]).sum())
    w1_error = float(_ORACLE.dgp.threshold_kw * math.exp(-_ORACLE.dgp.tail_rate) * below_mass)
    return value, w1_error


def tail_ppf_vec(dist, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    if dist.count_model == "poisson":
        pzero = math.exp(-dist.count_mean)
    else:
        odds = (1.0 - dist.nb_probability) / dist.nb_probability
        pzero = (1.0 + odds) ** (-dist.nb_size)
    below = p < pzero
    if np.any(below):
        vals, counts = np.unique(dist.baseline, return_counts=True)
        cdf = np.cumsum(counts / counts.sum())
        j = np.searchsorted(cdf, p[below] / pzero, side="left")
        out[below] = vals[np.minimum(j, len(vals) - 1)]
    if np.any(~below):
        pp = np.clip(p[~below], np.finfo(float).tiny, np.nextafter(1.0, 0.0))
        if dist.count_model == "poisson":
            g = np.clip(1.0 + np.log(pp) / dist.count_mean, 0.0, 1.0)
        else:
            odds = (1.0 - dist.nb_probability) / dist.nb_probability
            g = np.clip(1.0 - (pp ** (-1.0 / dist.nb_size) - 1.0) / odds, 0.0, 1.0)
        out[~below] = dist.threshold + stats.genpareto.ppf(g, dist.xi, loc=0.0, scale=dist.sigma)
    return out


def estimated_ppf_vec(engine, dist, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if isinstance(dist, engine.DiscreteDistribution):
        j = np.searchsorted(dist.cumulative, p, side="left")
        return dist.support[np.minimum(j, dist.support.size - 1)]
    if isinstance(dist, _AUDIT.NonnegativeGEV):
        rp = dist.f0 + p*dist.a
        return np.maximum(0.0, stats.genextreme.ppf(rp, dist.shape_c, loc=dist.loc, scale=dist.scale))
    if isinstance(dist, engine.TailProcessDistribution):
        return tail_ppf_vec(dist, p)
    if isinstance(dist, engine.NonnegativeGaussianKDE):
        lo = np.zeros_like(p)
        hi = np.full_like(p, float(dist.sample.max() + 12.0 * dist.bandwidth))
        # Vector bisection of the exact truncated Gaussian-mixture CDF. The
        # returned brackets imply a deterministic quantile error bound.
        for _ in range(44):
            mid = 0.5 * (lo + hi)
            cm = distribution_cdf_vec(engine, dist, mid)
            below = cm < p
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)
    raise TypeError(type(dist))


def continuous_w1(engine, dist) -> tuple[float, float]:
    # Tail-adapted quantile map p=1-u^r regularizes the integrable endpoint
    # singularity when either GEV/GPD shape is positive.
    tail_shape = max(0.15, float(getattr(dist, "xi", 0.0)), 0.0)
    r = min(25.0, 1.0/max(1e-6, 1.0-tail_shape))
    u = _GL_X
    survival = u**r
    weights = _GL_W*r*u**(r-1.0)
    qtrue = true_ppf_from_survival(survival)
    qhat = estimated_ppf_from_survival(engine, dist, survival)
    value = float(np.dot(weights, np.abs(qtrue-qhat)))
    if isinstance(dist, engine.NonnegativeGaussianKDE):
        # Bisection bracket width is bounded by the initial bracket / 2^44.
        qerr = float((dist.sample.max() + 12.0 * dist.bandwidth) / (2.0 ** 45))
    else:
        # Quantile formulas use IEEE double precision; this is a conservative
        # rounding-scale diagnostic, not a formal interval-arithmetic proof.
        qerr = float(64.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(qhat)))))
    return value, qerr


def tailprocess_w1(engine, dist) -> tuple[float, float]:
    """Exact atom integration below threshold plus tail-adapted GL above it."""
    baseline = np.asarray(dist.baseline, dtype=float)
    vals, counts = np.unique(baseline, return_counts=True)
    if dist.count_model == "poisson":
        pzero = math.exp(-dist.count_mean)
    else:
        odds=(1.0-dist.nb_probability)/dist.nb_probability
        pzero=(1.0+odds)**(-dist.nb_size)
    atom_law = engine.DiscreteDistribution(vals, counts/counts.sum())
    atom_law.probabilities = atom_law.probabilities*pzero
    atom_law.cumulative = np.cumsum(atom_law.probabilities)
    edges = np.r_[0.0, atom_law.cumulative]
    ftrue = true_cdf_vec(vals)
    cross = np.clip(ftrue, edges[:-1], edges[1:])
    klo = true_integrated_upper_quantile(edges[:-1])
    khi = true_integrated_upper_quantile(edges[1:])
    hc = true_stoploss_vec(vals)
    kc = (1.0-cross)*vals+hc
    kc = np.where(ftrue <= edges[:-1], klo, np.where(ftrue >= edges[1:], khi, kc))
    left = vals*(cross-edges[:-1])-(klo-kc)
    right = (kc-khi)-vals*(edges[1:]-cross)
    atom_value=float(np.maximum(left,0).sum()+np.maximum(right,0).sum())
    # The fitted continuous tail begins at pzero and is regularized at p=1.
    tail_shape=max(0.15,float(dist.xi),0.0)
    r=min(25.0,1.0/max(1e-6,1.0-tail_shape))
    u=_GL_X
    survival=(1.0-pzero)*u**r
    weights=_GL_W*r*(1.0-pzero)*u**(r-1.0)
    qtrue=true_ppf_from_survival(survival)
    qhat=estimated_ppf_from_survival(engine,dist,survival)
    tail_value=float(np.dot(weights,np.abs(qtrue-qhat)))
    below_mass=float(np.maximum(0.0,np.minimum(edges[1:],math.exp(-_ORACLE.dgp.tail_rate))-edges[:-1]).sum())
    below_err=float(_ORACLE.dgp.threshold_kw*math.exp(-_ORACLE.dgp.tail_rate)*below_mass)
    qerr=float(64*np.finfo(float).eps*max(1.0,float(np.max(np.abs(qhat)))))
    return atom_value+tail_value, below_err+qerr


def cdf_riemann_sup_upper(engine,dist,muhat:float,mutrue:float,tlo:float,thi:float,
                          n_contract:int,mean_eval_allowance:float=0.0) -> tuple[float,float,float]:
    """Conservative sup |Hhat-H| enclosure from monotone CDF endpoint bounds.

    Uses H(t)=mu-t+int_0^t F(s)ds. On each segment, monotonicity brackets
    int(Fhat-F); the final half-step uses |(Hhat-H)'|<=1. Reported roundoff
    allowance is explicit but is not directed-rounding interval arithmetic.
    """
    step=(thi-tlo)/int(n_contract)
    n_prefix=max(1,int(math.ceil(tlo/step)))
    prefix=np.linspace(0.0,tlo,n_prefix+1)
    contract=np.linspace(tlo,thi,int(n_contract)+1)
    x=np.r_[prefix,contract[1:]]
    fh=distribution_cdf_vec(engine,dist,x)
    ft=true_cdf_vec(x)
    dx=np.diff(x)
    low=np.cumsum((fh[:-1]-ft[1:])*dx)
    high=np.cumsum((fh[1:]-ft[:-1])*dx)
    # Contract grid nodes start at n_prefix; include prefix integral at tlo.
    idx=n_prefix+np.arange(int(n_contract)+1)
    lo=(muhat-mutrue)+low[idx-1]
    hi=(muhat-mutrue)+high[idx-1]
    cdf_allow=256*np.finfo(float).eps*max(1.0,thi)
    total_allow=float(mean_eval_allowance+2*thi*cdf_allow)
    upper=float(np.max(np.maximum(np.abs(lo),np.abs(hi)))+total_allow+0.5*step)
    # The point interval midpoint is a numerical estimate; it is not the bound.
    point=float(np.max(np.abs(0.5*(lo+hi))))
    riemann_width=float(np.max(hi-lo)+2*total_allow)
    return point,upper,riemann_width


def reconstruct_law(engine, law: dict) -> object:
    """Reconstruct only the law stored by the PASS no-upper audit; never refit."""
    kind = law["class"]
    if kind == "TailProcessDistribution":
        return engine.TailProcessDistribution(
            float(law["threshold_kw"]), float(law["xi"]), float(law["sigma_kw"]),
            np.asarray(law["baseline_monthly_maxima_kw"], dtype=float),
            str(law["count_model"]), float(law["count_mean"]), float(law["count_variance"]),
            None if law["nb_size"] is None else float(law["nb_size"]),
            None if law["nb_probability"] is None else float(law["nb_probability"]))
    if kind == "DiscreteDistribution":
        return engine.DiscreteDistribution(np.asarray(law["support_kw"], dtype=float),
                                           np.asarray(law["probabilities"], dtype=float))
    if kind == "NonnegativeGaussianKDE":
        dist = engine.NonnegativeGaussianKDE(np.asarray(law["sample_kw"], dtype=float))
        dist.bandwidth = float(law["bandwidth_kw"])
        dist._normalizers = special.ndtr(dist.sample/dist.bandwidth)
        dist._normalizer = float(dist._normalizers.sum())
        return dist
    if kind == "NonnegativeGEV":
        return _AUDIT.NonnegativeGEV(float(law["shape_c"]), float(law["loc_kw"]),
                                     float(law["scale_kw"]))
    raise ValueError(f"unsupported stored law class {kind!r}")


def solve_zeta(engine, dist, row) -> dict[str, float]:
    """Convex subgradient upper bound for saved D's fitted-objective gap."""
    alpha,kappa,rate=float(row.alpha),float(row.kappa),float(_SETTINGS["tariff"]["demand_rate"])
    s=float(row.S_kva); lo=float(_SETTINGS["tariff"]["delta"])*s
    hi=float(_SETTINGS["tariff"]["contract_upper_ratio"])*s
    d=float(np.clip(float(row.estimated_contract_D_kw),lo,hi)); x=alpha*d
    fm=distribution_cdf_left(engine,dist,x)
    fp=float(distribution_cdf_vec(engine,dist,np.array([x]))[0])
    slope_left=rate*(1+kappa*alpha*(fm-1.0))
    slope_right=rate*(1+kappa*alpha*(fp-1.0))
    if d<=lo+1e-10:
        gap_bound=0.0 if slope_right>=0 else abs(slope_right)*(hi-lo)
    elif d>=hi-1e-10:
        gap_bound=0.0 if slope_left<=0 else abs(slope_left)*(hi-lo)
    elif slope_left<=0<=slope_right:
        gap_bound=0.0
    elif slope_left>0:
        gap_bound=max(abs(slope_left),abs(slope_right))*(d-lo)
    else:
        gap_bound=max(abs(slope_left),abs(slope_right))*(hi-d)
    # Roundoff allowance for CDF and tariff arithmetic; no quadrature enters
    # this subgradient bound. It is a floating-point diagnostic, not interval arithmetic.
    zeta_roundoff=512*np.finfo(float).eps*max(1.0,rate*(d+kappa*alpha*d),rate*hi)
    zeta=float(gap_bound+zeta_roundoff)
    saved_cost=rate*(d+kappa*float(distribution_stoploss_vec(engine,dist,np.array([x]))[0]))
    pstar=1.0-1.0/(kappa*alpha)
    q=float(estimated_ppf_vec(engine,dist,np.array([pstar]))[0])
    quantile_d=float(np.clip(q/alpha,lo,hi))
    qdiag=abs(d-quantile_d)
    return {"solver_saved_vs_recomputed_abs_cny":abs(saved_cost-float(row.estimated_mode_cost_contract)),
            "solver_quantile_D_abs_kw":qdiag,
            "solver_observed_suboptimality_cny":float(gap_bound),
            "solver_zeta_bound_cny":zeta,
            "solver_zeta_roundoff_allowance_cny":float(zeta_roundoff),
            "solver_subgradient_left_cny_per_kw":float(slope_left),
            "solver_subgradient_right_cny_per_kw":float(slope_right)}


def process_chunk(task: tuple) -> dict[str, str|int|float]:
    fits_path, meta_path, audit_csv_path, primary_path, out_dir, start, end, expected_fit_hash, expected_csv_hash = task
    engine=_ENGINE
    out=Path(out_dir)
    if sha256(Path(fits_path))!=expected_fit_hash or sha256(Path(audit_csv_path))!=expected_csv_hash:
        raise AssertionError(f"stored fit/audit CSV hash mismatch for {start}:{end}")
    meta=json.loads(Path(meta_path).read_text(encoding="utf-8"))
    if meta.get("status")!="PASS" or meta.get("fits_sha256")!=expected_fit_hash or meta.get("csv_sha256")!=expected_csv_hash:
        raise AssertionError(f"stored audit chunk is not PASS/hash-aligned: {meta_path}")
    fits=[json.loads(line) for line in Path(fits_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(fits)!=(end-start)*5:
        raise AssertionError(f"expected {(end-start)*5} stored laws, found {len(fits)}")
    lawmap={(int(x["replication_id"]),str(x["method_id"])):x for x in fits}
    if any(x.get("support_policy")!="no_upper" or x.get("policy_kind")!="no_upper" or x.get("upper_B_kw") is not None for x in fits):
        raise AssertionError("fit JSONL contains a non-no-upper/upper-B law")
    audit=pd.read_csv(audit_csv_path)
    primary=pd.read_parquet(primary_path)
    if len(primary)!=(end-start)*45 or len(audit)!=(end-start)*45:
        raise AssertionError("primary/audit decisions have unexpected row count")
    keys=["replication_id","method_id","cell_id"]
    fields=["selected_mode","estimated_mean_kw","estimated_contract_D_kw","estimated_mode_cost_contract"]
    chk=primary[keys+fields].merge(audit[keys+fields],on=keys,suffixes=("_primary","_audit"),validate="one_to_one")
    if len(chk)!=len(primary): raise AssertionError("primary vs stored-audit join mismatch")
    if not np.array_equal(chk.selected_mode_primary.astype(str),chk.selected_mode_audit.astype(str)):
        raise AssertionError("primary vs stored-audit selected mode mismatch")
    for f in fields[1:]:
        err=np.max(np.abs(chk[f+"_primary"].to_numpy(float)-chk[f+"_audit"].to_numpy(float)))
        if err>2e-7: raise AssertionError(f"primary vs stored-audit {f} mismatch {err}")
    rows_dist=[]; rows_dec=[]; max_rebuilt_cost_diff=0.0
    for rep in range(start,end):
        dr=primary.loc[primary.replication_id==rep].copy()
        if len(dr)!=45: raise AssertionError(f"history {rep} has {len(dr)} decisions")
        dists={m:reconstruct_law(engine,lawmap[(rep,m)]["law"]) for m in ("TAIL","EVENT-EMP","GEV","KDE","EMP")}
        # The stored laws must reproduce primary means and fitted contract costs.
        means={m:float(d.mean()) for m,d in dists.items()}
        mean_reconstruction_error={}
        true_mu_internal=float(_ORACLE.mean())
        true_mu_ref=float(dr.true_mean_kw.iloc[0])
        oracle_mean_allowance=abs(true_mu_internal-true_mu_ref)+128*np.finfo(float).eps*max(1.0,abs(true_mu_ref))
        for method,dist in dists.items():
            mrows=dr.loc[dr.method_id==method]
            if len(mrows)!=9: raise AssertionError(f"{rep}/{method}: expected 9 cells")
            mean_diff=np.max(np.abs(mrows.estimated_mean_kw.to_numpy(float)-means[method]))
            if mean_diff>2e-5: raise AssertionError(f"stored law mean mismatch {rep}/{method}: {mean_diff}")
            mean_reconstruction_error[method]=float(mean_diff)
            if isinstance(dist,engine.DiscreteDistribution):
                w1,w1err=discrete_w1(engine,dist)
                wk="integrated_quantile_exact_for_discrete_law; oracle_stoploss_has_bounded_subthreshold_term"
            elif isinstance(dist,engine.TailProcessDistribution):
                w1,w1err=tailprocess_w1(engine,dist)
                wk=f"exact_atoms_plus_tail_adapted_Gauss_Legendre_{_W1_ORDER}; convergence_audit_separate"
            else:
                w1,w1err=continuous_w1(engine,dist)
                wk=f"tail_adapted_Gauss_Legendre_{_W1_ORDER}; convergence_audit_separate"
            w1_bounds=cdf_w1_riemann_bounds(engine,dist,n_grid=max(512,4*_W1_ORDER))
            rows_dist.append({"replication_id":rep,"method_id":method,"w1_kw":w1,
                              "w1_quantile_solver_error_bound_kw":w1err,
                              "w1_integration_error_certified":False,"w1_method":wk,
                              "w1_cdf_riemann_lower_kw":w1_bounds["lower"],
                              "w1_cdf_riemann_upper_kw":w1_bounds["upper"],
                              "w1_cdf_riemann_mesh_max_kw":w1_bounds["mesh_kw"],
                              "w1_union_tail_remainder_kw":w1_bounds["tail_kw"],
                              "w1_integration_limit_kw":w1_bounds["T_kw"],
                              "w1_cdf_riemann_fp_allowance_kw":w1_bounds["fp_allowance_kw"],
                              "estimated_mean_reconstructed_kw":means[method],
                              "w1_nonnegative_distributional_lower_kw":max(0.0,abs(means[method]-true_mu_ref)-mean_diff-oracle_mean_allowance),
                              "w1_nonnegative_distributional_upper_kw":means[method]+true_mu_ref+mean_diff+oracle_mean_allowance,
                              "w1_distributional_bound_is_coarse":True})
            d_costs=distribution_stoploss_vec(engine,dist,mrows.alpha.to_numpy(float)*mrows.estimated_contract_D_kw.to_numpy(float))
            cost_rebuilt=float(_SETTINGS["tariff"]["demand_rate"])*(mrows.estimated_contract_D_kw.to_numpy(float)+mrows.kappa.to_numpy(float)*d_costs)
            cost_diff=np.max(np.abs(cost_rebuilt-mrows.estimated_mode_cost_contract.to_numpy(float)))
            max_rebuilt_cost_diff=max(max_rebuilt_cost_diff,float(cost_diff))
            if cost_diff>0.05: raise AssertionError(f"stored law contract-cost mismatch {rep}/{method}: {cost_diff}")
        w1map={r["method_id"]:r for r in rows_dist[-5:]}
        for row in dr.itertuples(index=False):
            method=str(row.method_id); dist=dists[method]
            alpha,kappa=float(row.alpha),float(row.kappa)
            lo_d=float(_SETTINGS["tariff"]["delta"])*float(row.S_kva)
            hi_d=float(_SETTINGS["tariff"]["contract_upper_ratio"])*float(row.S_kva)
            tlo,thi=alpha*lo_d,alpha*hi_d
            grid=np.linspace(tlo,thi,_H_GRID+1)
            delta_h=distribution_stoploss_vec(engine,dist,grid)-true_stoploss_vec(grid)
            sup_point=float(np.max(np.abs(delta_h)))
            step=(thi-tlo)/_H_GRID
            mean_allowance=(mean_reconstruction_error[method]+oracle_mean_allowance+
                            256*np.finfo(float).eps*max(1.0,abs(means[method]),abs(true_mu_ref)))
            _,sup_grid_upper,sup_riemann_width=cdf_riemann_sup_upper(
                engine,dist,means[method],true_mu_ref,tlo,thi,_H_GRID,mean_allowance)
            zeta=solve_zeta(engine,dist,row)
            rate=float(_SETTINGS["tariff"]["demand_rate"])
            mean_err=means[method]-float(row.true_mean_kw)
            eps=max(rate*abs(mean_err),rate*kappa*sup_grid_upper)
            # Conservative distributional envelope: for nonnegative X,
            # 0<=H_X(t)<=E[X]. This avoids pretending the GL grid evaluations
            # are interval-certified; it is reported separately from the much
            # tighter numerically resolved primitive epsilon above.
            zeta_envelope=max(0.0,float(row.estimated_mode_cost_contract)-rate*lo_d)
            sup_h_envelope=max(float(row.true_mean_kw),means[method])
            epsilon_distributional_envelope=max(rate*abs(mean_err),rate*kappa*sup_h_envelope)
            w1rec=w1map[method]
            w1point=float(w1rec["w1_kw"]); w1upper=float(w1rec["w1_cdf_riemann_upper_kw"])
            uniform_radius=2*rate*max(1.0,kappa)*w1upper+zeta["solver_zeta_bound_cny"]
            uniform_point_radius=2*rate*max(1.0,kappa)*w1point+zeta["solver_zeta_bound_cny"]
            mode_key=lambda x:("contract" if "contract" in str(x).lower() else
                               "capacity" if "capacity" in str(x).lower() else
                               "actual" if "actual" in str(x).lower() else "unknown")
            oracle_key=mode_key(row.oracle_mode)
            mode_radii={"capacity":0.0,"actual":rate*w1upper,"contract":rate*kappa*w1upper}
            point_mode_radii={"capacity":0.0,"actual":rate*w1point,"contract":rate*kappa*w1point}
            pair_radius=(mode_radii.get(oracle_key,rate*max(1.0,kappa)*w1upper)+
                         max((v for m,v in mode_radii.items() if m!=oracle_key),default=0.0)+
                         zeta["solver_zeta_bound_cny"])
            pair_point_radius=(point_mode_radii.get(oracle_key,rate*max(1.0,kappa)*w1point)+
                         max((v for m,v in point_mode_radii.items() if m!=oracle_key),default=0.0)+
                         zeta["solver_zeta_bound_cny"])
            actual_err=abs(float(row.delta_actual_cny)); contract_err=abs(float(row.delta_contract_cny))
            if eps+zeta["solver_zeta_bound_cny"]+2e-4<max(actual_err,contract_err):
                raise AssertionError(f"primitive epsilon underbounds value error {rep}/{method}/{row.cell_id}: {eps} < {max(actual_err,contract_err)}")
            regret=float(row.regret_cny_per_month)
            full_bound=2*eps+zeta["solver_zeta_bound_cny"]
            contract_correct=(oracle_key=="contract" and mode_key(row.selected_mode)=="contract")
            contract_bound=2*rate*kappa*sup_grid_upper+zeta["solver_zeta_bound_cny"] if contract_correct else np.nan
            rows_dec.append({"replication_id":rep,"method_id":method,"cell_id":str(row.cell_id),
                "policy_id":str(row.policy_id),"oracle_mode":str(row.oracle_mode),"selected_mode":str(row.selected_mode),
                "action_correct":bool(row.action_correct),"oracle_margin_cny":float(row.oracle_margin_cny),
                "mean_error_kw_signed":float(mean_err),"w1_kw":w1point,
                "w1_cdf_riemann_lower_kw":float(w1rec["w1_cdf_riemann_lower_kw"]),
                "w1_cdf_riemann_upper_kw":w1upper,
                "w1_point_certificate_uniform_diagnostic":bool(float(row.oracle_margin_cny)>uniform_point_radius),
                "w1_certificate_uniform_upper":bool(float(row.oracle_margin_cny)>uniform_radius),
                "w1_certificate_uniform_violation":bool(float(row.oracle_margin_cny)>uniform_radius and not bool(row.action_correct)),
                "w1_radius_uniform_cny":uniform_radius,
                "w1_point_certificate_pairwise_diagnostic":bool(float(row.oracle_margin_cny)>pair_point_radius),
                "w1_certificate_pairwise_upper":bool(float(row.oracle_margin_cny)>pair_radius),
                "w1_certificate_pairwise_violation":bool(float(row.oracle_margin_cny)>pair_radius and not bool(row.action_correct)),
                "w1_radius_pairwise_cny":pair_radius,
                "sup_H_point_estimate_kw":sup_point,"sup_H_cdf_riemann_upper_kw":sup_grid_upper,
                "sup_H_riemann_interval_width_kw":sup_riemann_width,
                "sup_H_grid_addon_kw":0.5*step,"sup_H_mean_eval_allowance_kw":mean_allowance,
                "sup_H_uses_monotone_cdf_endpoint_bounds":True,
                "sup_H_direct_stoploss_estimate_quadrature_order":_H_GL_X.size,
                "primitive_epsilon_cny":eps,"primitive_epsilon_cdf_riemann_numerical_bound_cny":eps,
                "primitive_epsilon_has_directed_rounding":False,"actual_value_abs_error_cny":actual_err,
                "primitive_epsilon_distributional_envelope_cny":epsilon_distributional_envelope,
                "sup_H_distributional_envelope_kw":sup_h_envelope,
                "solver_zeta_distributional_envelope_cny":zeta_envelope,
                "contract_value_abs_error_cny":contract_err,
                "solver_saved_vs_recomputed_abs_cny":zeta["solver_saved_vs_recomputed_abs_cny"],
                "solver_quantile_D_abs_kw":zeta["solver_quantile_D_abs_kw"],
                "solver_observed_suboptimality_cny":zeta["solver_observed_suboptimality_cny"],
                "solver_zeta_bound_cny":zeta["solver_zeta_bound_cny"],
                "primitive_gap_certificate_numerical":bool(float(row.oracle_margin_cny)>2*eps+zeta["solver_zeta_bound_cny"]),
                "primitive_certificate_violation_numerical":bool(float(row.oracle_margin_cny)>2*eps+zeta["solver_zeta_bound_cny"] and not bool(row.action_correct)),
                "primitive_gap_certificate_distributional_envelope":bool(float(row.oracle_margin_cny)>2*epsilon_distributional_envelope+zeta_envelope),
                "primitive_certificate_violation_distributional_envelope":bool(float(row.oracle_margin_cny)>2*epsilon_distributional_envelope+zeta_envelope and not bool(row.action_correct)),
                "full_regret_cny":regret,"full_regret_bound_2epsilon_plus_zeta_cny":full_bound,
                "full_regret_bound_violation":bool(regret>full_bound+1e-4),
                "oracle_and_selected_contract":contract_correct,"correct_contract_regret_bound_cny":contract_bound,
                "correct_contract_regret_bound_violation":bool(contract_correct and regret>contract_bound+1e-4)})
    stem=f"{start:05d}_{end-1:05d}"
    dp=out/f"primitive_distances_{stem}.parquet"; cp=out/f"primitive_decisions_{stem}.parquet"
    pd.DataFrame(rows_dist).to_parquet(dp,index=False); pd.DataFrame(rows_dec).to_parquet(cp,index=False)
    return {"start":start,"end":end,"dist_rows":len(rows_dist),"decision_rows":len(rows_dec),
            "dist_path":str(dp),"decision_path":str(cp),"fits_sha256":expected_fit_hash,
            "audit_csv_sha256":expected_csv_hash,"max_rebuilt_contract_cost_abs_diff_cny":max_rebuilt_cost_diff,
            "dist_sha256":sha256(dp),"decision_sha256":sha256(cp),
            "status":"PASS"}


def write_summaries(out:Path,manifest:dict) -> dict[str,str]:
    """Write compact tables/README after all hash-verified chunks pass."""
    dist=pd.concat([pd.read_parquet(x["dist_path"]) for x in manifest["tasks"]],ignore_index=True)
    dec=pd.concat([pd.read_parquet(x["decision_path"]) for x in manifest["tasks"]],ignore_index=True)
    w1=(dist.groupby("method_id",as_index=False)
        .agg(n_histories=("w1_kw","size"),w1_mean_kw=("w1_kw","mean"),w1_median_kw=("w1_kw","median"),
             w1_p90_kw=("w1_kw",lambda s:s.quantile(.9)),w1_p99_kw=("w1_kw",lambda s:s.quantile(.99)),
             cdf_lower_mean_kw=("w1_cdf_riemann_lower_kw","mean"),cdf_upper_mean_kw=("w1_cdf_riemann_upper_kw","mean"),
             cdf_interval_width_mean_kw=("w1_cdf_riemann_upper_kw",lambda s:float((s-dist.loc[s.index,"w1_cdf_riemann_lower_kw"]).mean())),
             tail_remainder_mean_kw=("w1_union_tail_remainder_kw","mean"),
             quadrature_point_is_certified=("w1_integration_error_certified","all")))
    w1.to_csv(out/"w1_summary_by_method.csv",index=False)
    grouping=["method_id","policy_id","cell_id","oracle_mode"]
    summary=(dec.groupby(grouping,as_index=False)
        .agg(n=("primitive_epsilon_cny","size"),
             epsilon_median_cny=("primitive_epsilon_cny","median"),epsilon_p90_cny=("primitive_epsilon_cny",lambda s:s.quantile(.9)),
             strict_distributional_epsilon_median_cny=("primitive_epsilon_distributional_envelope_cny","median"),
             supH_point_median_kw=("sup_H_point_estimate_kw","median"),supH_Riemann_upper_median_kw=("sup_H_cdf_riemann_upper_kw","median"),
             supH_Riemann_upper_p90_kw=("sup_H_cdf_riemann_upper_kw",lambda s:s.quantile(.9)),
             supH_Riemann_interval_width_median_kw=("sup_H_riemann_interval_width_kw","median"),
             w1_point_mean_kw=("w1_kw","mean"),w1_CDF_upper_mean_kw=("w1_cdf_riemann_upper_kw","mean"),
             w1_CDF_lower_mean_kw=("w1_cdf_riemann_lower_kw","mean"),
             signed_mean_error_mean_kw=("mean_error_kw_signed","mean"),signed_mean_error_median_kw=("mean_error_kw_signed","median"),
             solver_zeta_median_cny=("solver_zeta_bound_cny","median"),solver_zeta_p99_cny=("solver_zeta_bound_cny",lambda s:s.quantile(.99)),
             numerical_certificate_coverage=("primitive_gap_certificate_numerical","mean"),
             numerical_certificate_violations=("primitive_certificate_violation_numerical","sum"),
             distributional_envelope_coverage=("primitive_gap_certificate_distributional_envelope","mean"),
             distributional_envelope_violations=("primitive_certificate_violation_distributional_envelope","sum"),
             W1_uniform_upper_coverage=("w1_certificate_uniform_upper","mean"),
             W1_uniform_upper_violations=("w1_certificate_uniform_violation","sum"),
             W1_pairwise_upper_coverage=("w1_certificate_pairwise_upper","mean"),
             W1_pairwise_upper_violations=("w1_certificate_pairwise_violation","sum"),
             full_regret_bound_violations=("full_regret_bound_violation","sum"),
             correct_contract_n=("oracle_and_selected_contract","sum"),
             correct_contract_regret_bound_violations=("correct_contract_regret_bound_violation","sum")))
    summary.to_csv(out/"primitive_summary_by_method_cell_oracle.csv",index=False)
    validation=pd.DataFrame([{k:r.get(k) for k in ("start","end","status","fits_sha256","audit_csv_sha256",
                  "dist_sha256","decision_sha256","max_rebuilt_contract_cost_abs_diff_cny","dist_rows","decision_rows")}
                  for r in manifest["tasks"]])
    validation.to_csv(out/"validation_summary_by_chunk.csv",index=False)
    overall={"status":"PASS","n_history_laws":int(len(dist)),"n_decision_rows":int(len(dec)),
        "numerical_certificate_coverage":float(dec.primitive_gap_certificate_numerical.mean()),
        "numerical_certificate_violations":int(dec.primitive_certificate_violation_numerical.sum()),
        "distributional_envelope_coverage":float(dec.primitive_gap_certificate_distributional_envelope.mean()),
        "distributional_envelope_violations":int(dec.primitive_certificate_violation_distributional_envelope.sum()),
        "w1_uniform_upper_coverage":float(dec.w1_certificate_uniform_upper.mean()),
        "w1_uniform_upper_violations":int(dec.w1_certificate_uniform_violation.sum()),
        "w1_pairwise_upper_coverage":float(dec.w1_certificate_pairwise_upper.mean()),
        "w1_pairwise_upper_violations":int(dec.w1_certificate_pairwise_violation.sum()),
        "full_regret_bound_violations":int(dec.full_regret_bound_violation.sum()),
        "correct_contract_rows":int(dec.oracle_and_selected_contract.sum()),
        "correct_contract_regret_bound_violations":int(dec.correct_contract_regret_bound_violation.sum()),
        "max_actual_or_contract_value_error_cny":float(np.maximum(dec.actual_value_abs_error_cny,dec.contract_value_abs_error_cny).max()),
        "max_primitive_epsilon_cny":float(dec.primitive_epsilon_cny.max()),
        "max_distributional_envelope_epsilon_cny":float(dec.primitive_epsilon_distributional_envelope_cny.max()),
        "max_solver_zeta_cny":float(dec.solver_zeta_bound_cny.max()),
        "w1_point_estimate_inside_cdf_riemann_bracket_n":int(((dist.w1_kw>=dist.w1_cdf_riemann_lower_kw-1e-6)&(dist.w1_kw<=dist.w1_cdf_riemann_upper_kw+1e-6)).sum()),
        "w1_point_estimate_outside_cdf_riemann_bracket_n":int(((dist.w1_kw<dist.w1_cdf_riemann_lower_kw-1e-6)|(dist.w1_kw>dist.w1_cdf_riemann_upper_kw+1e-6)).sum())}
    (out/"primitive_bounds_summary.json").write_text(json.dumps(overall,indent=2),encoding="utf-8")
    readme="""# Primitive bounds and distribution distances

This is independent post-processing of the corrected 10,000-history no-upper primary run. It reconstructs the five laws from PASS stored fit JSONL files; no laws are refit and no event/monthly histories are read. The manifest records source and output chunk SHA256 hashes.

## Definitions and numerical status

`w1_kw` is the tail-adapted Gauss-Legendre estimate of W1(F,Fhat)=integral_0^1 |Q_F(p)-Q_Fhat(p)| dp. Separate `w1_cdf_riemann_lower_kw` and `w1_cdf_riemann_upper_kw` fields enclose the finite-grid CDF integral using monotonic endpoint bounds. Every empirical atom and stored tail threshold is included. The remaining infinite tail is bounded analytically: count-process union bounds for Poisson/NB GPD maxima, the GEV survival bound 1-exp(-v)<=v, the analytic KDE stop-loss, or zero past finite empirical support. Tail remainder, mesh, integration limit, and floating-point allowance are stored. Double-precision SciPy CDF values and arithmetic are not directed-rounding interval arithmetic; the GL point estimate has no certified quadrature error.

For nonnegative laws, H_F(t)=E[(X-t)^+]=mu_F-t+integral_0^t F(s) ds. `sup_H_cdf_riemann_upper_kw` uses monotone endpoint Riemann bounds for both CDFs and |d(Hhat-H)/dt|<=1 between tariff-grid nodes. Mean reconstruction differences are included. `sup_H_point_estimate_kw` is a separate stop-loss/quadrature point estimate.

`solver_zeta_bound_cny` bounds the fitted convex contract objective gap at saved D using its left/right CDF subgradient, feasible distance, boundary normal cones, and atom-aware subgradients. The primitive radius is epsilon=max(rd*abs(muhat-mu), rd*kappa*sup_H_upper); the mode certificate uses oracle_margin > 2*epsilon + zeta. W1 uniform and pair-specific upper certificates use the CDF W1 upper endpoint; every pairwise radius includes one zeta allowance for the contract competitor, regardless of the oracle mode. Point-estimate versions are diagnostic. The outputs also check full regret R<=2*epsilon+zeta and, where both oracle and selected modes are contract, correct-contract regret R<=2*rd*kappa*sup_H_upper+zeta.

`primitive_epsilon_distributional_envelope_cny` is a coarser nonnegative-law envelope based on 0<=H<=mu and a feasible-cost lower bound. It is reported separately. The CDF Riemann and tail calculations are numerically guarded enclosures with explicit floating-point allowances, not directed-rounding interval arithmetic; report them as numerical bounds with the stated allowance, not machine-certified intervals. Violations remain in the output and are counted.

## Files

- `w1_summary_by_method.csv`: W1 point estimates and CDF enclosures by method.
- `primitive_summary_by_method_cell_oracle.csv`: W1, epsilon, sup-H, solver gap, regret checks, and certificate coverage by method, tariff cell, and oracle mode.
- `validation_summary_by_chunk.csv`: stored input/output hashes, row counts, and maximum decision-cost reconstruction discrepancy for each chunk.
- `primitive_bounds_summary.json`: total row counts, coverage, violations, and global maxima.
"""
    (out/"README.md").write_text(readme,encoding="utf-8")
    return {name:sha256(out/name) for name in ("w1_summary_by_method.csv","primitive_summary_by_method_cell_oracle.csv",
             "validation_summary_by_chunk.csv","primitive_bounds_summary.json","README.md")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-repo", type=Path, required=True)
    parser.add_argument("--audit-script", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--decision-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-replications", type=int, default=10000)
    parser.add_argument("--h-grid", type=int, default=512)
    parser.add_argument("--h-quad", type=int, default=64)
    parser.add_argument("--w1-order", type=int, default=128)
    parser.add_argument("--resume", action="store_true", help="resume only hash-verified completed chunks")
    args = parser.parse_args()
    if not 1 <= args.workers <= 15:
        raise ValueError("worker cap is 15, matching the user-authorized local CPU budget")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    am_path=args.audit_root/"audit_manifest.json"
    if not am_path.is_file(): raise FileNotFoundError(am_path)
    am=json.loads(am_path.read_text(encoding="utf-8"))
    if am.get("status")!="PASS" or am.get("range")!=[0,10000] or am.get("policies")!=["no_upper"]:
        raise AssertionError("stored no-upper audit manifest must be PASS for range [0,10000]")
    source_hash=sha256(args.decision_file)
    expected_primary_hash=am.get("merged_decisions_sha256")
    if not expected_primary_hash:
        raise AssertionError("stored audit manifest has no merged decision hash")
    if source_hash!=expected_primary_hash:
        raise AssertionError(f"primary decision hash mismatch: {source_hash}")
    dec=pd.read_csv(args.decision_file)
    if dec.replication_id.nunique()<args.max_replications:
        raise AssertionError("decision table does not contain requested complete histories")
    tmp=args.out/"_scratch_primary_chunks"
    tmp.mkdir(exist_ok=True)
    tasks = []
    chunks=am["chunks"]
    for ch in chunks:
        if ch.get("status")!="PASS": raise AssertionError("a stored fit chunk is not PASS")
        start,end=int(ch["start"]),int(ch["end"])
        if start>=args.max_replications: continue
        if end>args.max_replications:
            if args.max_replications!=50:
                raise ValueError("partial fit-chunk runs are allowed only for the 50-history validation pilot")
            end=args.max_replications
        stem=f"rep_{int(ch['start']):05d}_{int(ch['end'])-1:05d}"
        fits=args.audit_root/f"{stem}_fits.jsonl"
        meta=args.audit_root/f"{stem}_meta.json"
        audit_csv=args.audit_root/f"{stem}.csv"
        if not fits.is_file() or not meta.is_file() or not audit_csv.is_file():
            raise FileNotFoundError(f"incomplete stored fit chunk {stem}")
        meta_obj=json.loads(meta.read_text(encoding="utf-8"))
        if meta_obj.get("status")!="PASS" or meta_obj.get("start")!=int(ch["start"]) or meta_obj.get("end")!=int(ch["end"]):
            raise AssertionError(f"stored meta mismatch for {stem}")
        if sha256(fits)!=meta_obj.get("fits_sha256") or sha256(audit_csv)!=meta_obj.get("csv_sha256"):
            raise AssertionError(f"stored audit file hash failure for {stem}")
        sub=dec[(dec.replication_id>=start)&(dec.replication_id<end)]
        if len(sub)!=(end-start)*45: raise AssertionError(f"incomplete primary decisions in {start}:{end}")
        dpath=tmp/f"primary_{start:05d}_{end-1:05d}.parquet"
        sub.to_parquet(dpath,index=False)
        tasks.append((str(fits),str(meta),str(audit_csv),str(dpath),str(args.out),start,end,
                      meta_obj["fits_sha256"],meta_obj["csv_sha256"]))
    if sum(t[6]-t[5] for t in tasks)!=args.max_replications:
        raise AssertionError("stored fits do not cover requested replication range exactly")
    manifest_path=args.out/"primitive_bounds_manifest.json"
    code_hash=sha256(Path(__file__))
    config={"max_replications":args.max_replications,"h_grid_intervals_per_cell":args.h_grid,
            "w1_gauss_legendre_order":args.w1_order,"h_tail_quadrature_order":args.h_quad,
            "source_decision_sha256":source_hash,"stored_audit_manifest_sha256":sha256(am_path),
            "stored_audit_signature":am.get("signature"),
            "source_base_engine_sha256":sha256(args.base_repo/"scripts"/"run_revision_experiment.py"),
            "source_audit_class_sha256":sha256(args.audit_script),"compute_script_sha256":code_hash}
    existing=[]
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"{manifest_path} exists; use --resume or choose a new output directory")
        old=json.loads(manifest_path.read_text(encoding="utf-8"))
        for k,v in config.items():
            if old.get(k)!=v: raise AssertionError(f"resume refused: manifest {k} mismatch")
        existing=list(old.get("tasks",[]))
    elif args.resume:
        raise FileNotFoundError("--resume requested but no primitive_bounds_manifest.json exists")
    # A completed chunk is reusable only if both output hashes and row counts verify.
    valid={}
    for rec in existing:
        if rec.get("status")!="PASS": continue
        dp,cp=Path(rec.get("dist_path","")),Path(rec.get("decision_path",""))
        if not dp.is_file() or not cp.is_file(): continue
        if sha256(dp)!=rec.get("dist_sha256") or sha256(cp)!=rec.get("decision_sha256"): continue
        if len(pd.read_parquet(dp))!=rec.get("dist_rows") or len(pd.read_parquet(cp))!=rec.get("decision_rows"): continue
        valid[(int(rec["start"]),int(rec["end"]))]=rec
    pending=[t for t in tasks if (int(t[5]),int(t[6])) not in valid]
    manifest={"status":"RUNNING",**config,"workers":args.workers,
              "stored_law_source":"PASS support-policy audit fit JSONL; no refit or history/event input",
              "tasks":list(valid.values()),"started_unix":time.time(),
              "resumed":bool(args.resume),"resume_verified_chunks":len(valid),"planned_tasks":len(tasks)}
    write_json_atomic(manifest_path,manifest)
    try:
      with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                               initargs=(str(args.base_repo),str(args.audit_script),args.w1_order,args.h_grid,args.h_quad)) as pool:
        futs={pool.submit(process_chunk,t):t for t in pending}
        for fut in as_completed(futs):
            task = futs[fut]
            try:
                record = fut.result()
            except Exception as exc:
                manifest["status"] = "FAIL"
                manifest["failure"] = {"task": task[-2:], "error": repr(exc), "traceback": traceback.format_exc()}
                write_json_atomic(manifest_path,manifest)
                raise
            manifest["tasks"].append(record)
            manifest["completed_tasks"]=len(manifest["tasks"])
            write_json_atomic(manifest_path,manifest)
    except Exception:
      raise
    manifest["tasks"].sort(key=lambda x: x["start"])
    if len(manifest["tasks"])!=len(tasks):
        manifest["status"]="RUNNING"
        manifest["completed_tasks"]=len(manifest["tasks"])
        write_json_atomic(manifest_path,manifest)
        raise RuntimeError("not all output chunks are complete; rerun with --resume")
    manifest["status"] = "PASS"
    manifest["completed_tasks"] = len(manifest["tasks"])
    manifest["elapsed_seconds"] = time.time() - started
    manifest["summary_files_sha256"]=write_summaries(args.out,manifest)
    write_json_atomic(manifest_path,manifest)
    print(json.dumps({"status": manifest["status"], "tasks": len(tasks),
                      "elapsed_seconds": manifest["elapsed_seconds"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
