#!/usr/bin/env python3
"""Audit and summarize the fresh 10,000-history extension outputs."""
from __future__ import annotations
import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import t as student_t

PAIR_TIE_TOL_CNY = 1e-8
REGRET_RECON_TOL_CNY = 1e-5
ES_VALUE_EQUALITY_TOL_CNY = 1e-5
ACTION_FEASIBILITY_TOL_KW = 1e-7


def load_engine(base_repo: Path):
    sys.path.insert(0, str(base_repo / 'src'))
    spec = importlib.util.spec_from_file_location('base_revision_engine', base_repo / 'scripts/run_revision_experiment.py')
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod


def mean_ci(x: pd.Series) -> tuple[float, float, float, float]:
    values = x.to_numpy(dtype=float)
    n = len(values); mean = float(values.mean()); sd = float(values.std(ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    half = float(student_t.ppf(.975, n-1) * se) if n > 1 else np.nan
    return mean, sd, se, half


def empirical_upper_es(values: np.ndarray, probability: float) -> float:
    """RU upper expected shortfall of an equally weighted finite sample.

    The tail mass is split at its boundary observation, so this remains exact
    when n*(1-p) is non-integer or the empirical distribution has atoms.
    """
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or x.size==0 or not np.isfinite(x).all():
        raise ValueError('values must be a finite nonempty vector')
    if not 0.0<=probability<1.0:
        raise ValueError('probability must lie in [0,1)')
    mass=(1.0-probability)*x.size
    k=int(np.floor(mass+1e-12))
    fraction=mass-k
    ordered=np.sort(x)
    total=float(ordered[-k:].sum()) if k else 0.0
    if fraction>1e-12:
        total+=fraction*float(ordered[-k-1])
    return total/mass


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float,float]:
    if n == 0: return np.nan, np.nan
    p = k/n; den = 1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half, center+half


def l_inf_region_radius(costs: dict[str,float], oracle_mode: str, target: str) -> float:
    """Distance from zero (delta_actual, delta_contract) to target's closed region."""
    a, c, q = costs['actual_maximum_demand'], costs['contracted_maximum_demand'], costs['capacity']
    # x = delta_actual; y = delta_contract; target-mode estimated value must
    # be no greater than either rival. Capacity perturbation is exactly zero.
    inequalities = {
        'actual_maximum_demand': [(1,0,q-a), (1,-1,c-a)],
        'capacity': [(-1,0,a-q), (0,-1,c-q)],
        'contracted_maximum_demand': [(0,1,q-c), (-1,1,a-c)],
    }
    rows=[]; rhs=[]
    for u,v,b in inequalities[target]: rows.append([u,v,0]); rhs.append(b)
    rows.extend([[1,0,-1],[-1,0,-1],[0,1,-1],[0,-1,-1]])
    rhs.extend([0,0,0,0])
    result=linprog([0,0,1],A_ub=np.asarray(rows,float),b_ub=np.asarray(rhs,float),
                   bounds=[(None,None),(None,None),(0,None)],method='highs')
    if not result.success: raise RuntimeError(f'LP failed for {oracle_mode}->{target}: {result.message}')
    return float(result.fun)


def pairwise_order_audit(oracle_difference: float, estimated_differences: np.ndarray,
                         tie_tol: float = PAIR_TIE_TOL_CNY) -> dict[str, float | int | bool]:
    """Separate exact engine ties, strict crossings, and numerical near-ties."""
    estimated_differences=np.asarray(estimated_differences,dtype=float)
    oracle_pair_tied=float(oracle_difference)==0.0
    oracle_pair_near_tied=(not oracle_pair_tied and abs(float(oracle_difference))<=tie_tol)
    estimated_tied=estimated_differences==0.0
    estimated_near_tied=(~estimated_tied)&(np.abs(estimated_differences)<=tie_tol)
    if oracle_pair_tied:
        # A pair tied under the oracle has no strict order to reverse.
        reversals=np.nan; reversal_frequency=np.nan
        positive=int((estimated_differences>0.0).sum())
        negative=int((estimated_differences < 0.0).sum())
    else:
        oracle_sign=np.sign(oracle_difference)
        reversals=int((np.sign(estimated_differences[~estimated_tied]) == -oracle_sign).sum())
        reversal_frequency=reversals/len(estimated_differences)
        positive=np.nan; negative=np.nan
    return {'oracle_pair_tied_exact':bool(oracle_pair_tied),
            'oracle_pair_near_tied_atol':bool(oracle_pair_near_tied),
            'crossed_pairwise_boundary_count':reversals,
            'crossed_pairwise_boundary_frequency':reversal_frequency,
            'estimated_pair_exact_tie_count':int(estimated_tied.sum()),
            'estimated_pair_exact_tie_frequency':float(estimated_tied.mean()),
            'estimated_pair_near_tie_count':int(estimated_near_tied.sum()),
            'estimated_pair_near_tie_frequency':float(estimated_near_tied.mean()),
            'oracle_tie_estimated_positive_side_count':positive,
            'oracle_tie_estimated_negative_side_count':negative,
            'pairwise_tie_diagnostic_tolerance_cny':float(tie_tol)}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-repo',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True,help='Directory containing raw/primary_decisions_10000.csv.gz')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--expected-replications',type=int,default=10000)
    parser.add_argument('--tail-bootstrap',type=int,default=500,help='history-level bootstrap draws for regret quantile uncertainty')
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    engine=load_engine(args.base_repo)
    settings=engine.load_settings(); dgp=engine.dgp_from_settings(settings)
    oracle=engine.OracleDistribution(dgp)
    cells=engine.policies_and_cells(settings,oracle,False)
    mode_names=[engine.CAPACITY,engine.ACTUAL,engine.CONTRACT]
    oracle_rows=[]; oracle_by_cell={}
    for c in cells:
        m=c['oracle']['mode_costs']; b=c['oracle']['selected_mode']; v=float(c['oracle']['best_value_cny'])
        if not np.isfinite([*m.values(), v, float(c['oracle']['margin_cny'])]).all():
            raise AssertionError(f"nonfinite oracle mode value or margin in {c['cell_id']}")
        rec={'cell_id':c['cell_id'],'alpha':c['policy'].alpha,'utilization':c['utilization'],
             'S_kva':c['S_kva'],'oracle_mode':b,'oracle_best_value_cny':v,
             'oracle_gap_cny':c['oracle']['margin_cny']}
        for k in mode_names: rec['oracle_value_'+k+'_cny']=float(m[k])
        # This tolerance defines a numerical near-minimizer set. The base
        # engine itself resolves only exact equal costs by the mode-name tie rule.
        rec['oracle_near_minimizer_set_atol']='|'.join(
            k for k in mode_names if abs(float(m[k])-v)<=PAIR_TIE_TOL_CNY
        )
        rec['oracle_near_minimizer_count_atol']=sum(
            abs(float(m[k])-v)<=PAIR_TIE_TOL_CNY for k in mode_names
        )
        rec['oracle_near_minimizer_tolerance_cny']=PAIR_TIE_TOL_CNY
        rec['oracle_tie_rule']='min((mode_cost, mode_name)); names: actual_maximum_demand < capacity < contracted_maximum_demand'
        oracle_rows.append(rec); oracle_by_cell[c['cell_id']]=c
    pd.DataFrame(oracle_rows).to_csv(args.out/'oracle_mode_values_9_cells.csv',index=False)

    raw_path=args.data/'raw/primary_decisions_10000.csv.gz'
    df=pd.read_csv(raw_path)
    expected=args.expected_replications*5*9
    key=['replication_id','method_id','cell_id']
    if len(df)!=expected: raise AssertionError(f'expected {expected} decisions, found {len(df)}')
    if df.duplicated(key).any(): raise AssertionError('duplicate decision keys')
    if df[key].isna().any().any(): raise AssertionError('missing primary keys')
    if set(df.replication_id.unique())!=set(range(args.expected_replications)):
        raise AssertionError('decision table does not contain the complete configured history-ID range')
    if set(df.method_id.unique())!={'TAIL','EVENT-EMP','GEV','KDE','EMP'}:
        raise AssertionError(f'unexpected estimator-method domain: {sorted(df.method_id.unique())}')
    expected_cells={c['cell_id'] for c in cells}
    if set(df.cell_id.unique())!=expected_cells:
        raise AssertionError(f'unexpected tariff-cell domain: {sorted(df.cell_id.unique())}')
    combo_sizes=df.groupby(['method_id','cell_id'],observed=True).size()
    if len(combo_sizes)!=5*9 or not (combo_sizes==args.expected_replications).all():
        raise AssertionError('each estimator-method × tariff-cell group must contain exactly the expected history count')
    numeric_gate_cols=[
        'estimated_mode_cost_capacity','estimated_mode_cost_actual','estimated_mode_cost_contract',
        'estimated_contract_D_kw','true_mean_kw','true_q99_kw','true_stoploss_kw',
        'stoploss_threshold_kw','oracle_contract_quantile_p_star_kw','oracle_es_p_star_kw',
        'estimated_contract_quantile_p_star_kw','estimated_es_p_star_kw',
    ]
    finite_gate=np.isfinite(df[numeric_gate_cols].to_numpy(dtype=float))
    if not finite_gate.all():
        bad=np.argwhere(~finite_gate)[0]
        raise AssertionError(f'nonfinite value in {numeric_gate_cols[int(bad[1])]} at raw row {int(bad[0])}')
    oracle_lookup={r['cell_id']:r for r in oracle_rows}
    expected_oracle_mode=df.cell_id.map({c:r['oracle_mode'] for c,r in oracle_lookup.items()})
    if not (df.oracle_mode.astype(str).to_numpy()==expected_oracle_mode.astype(str).to_numpy()).all():
        raise AssertionError('raw oracle_mode does not match the oracle recomputed from this analyzer config')
    for observed, expected_values, label in [
        ('oracle_best_value_cny', [oracle_lookup[c]['oracle_best_value_cny'] for c in df.cell_id], 'oracle best value'),
        ('oracle_margin_cny', [oracle_lookup[c]['oracle_gap_cny'] for c in df.cell_id], 'oracle mode gap'),
        ('alpha', [oracle_lookup[c]['alpha'] for c in df.cell_id], 'alpha'),
        ('utilization', [oracle_lookup[c]['utilization'] for c in df.cell_id], 'utilization'),
        ('S_kva', [oracle_lookup[c]['S_kva'] for c in df.cell_id], 'reference capacity'),
    ]:
        if not np.allclose(df[observed].to_numpy(dtype=float), np.asarray(expected_values,dtype=float),
                           rtol=1e-12,atol=REGRET_RECON_TOL_CNY if 'value' in observed or 'gap' in observed else 1e-10):
            raise AssertionError(f'raw {label} values do not match recomputed oracle/config')
    # Independently reconstruct the oracle-side functionals saved on every row.
    # This catches a stale or mismatched truth table before using signed errors.
    truth_rows=[]
    for c in cells:
        policy=c['policy']; cell_id=c['cell_id']
        contract_d=float(c['oracle']['contract_D_kw'])
        threshold=float(policy.alpha*contract_d)
        p_star=float(policy.contract_probability)
        q_star=float(oracle.ppf(p_star))
        es_star=float(q_star+policy.kappa*policy.alpha*oracle.stop_loss(q_star))
        expected_truth={
            'true_mean_kw':float(oracle.mean()),
            'true_q99_kw':float(oracle.ppf(.99)),
            'stoploss_threshold_kw':threshold,
            'true_stoploss_kw':float(oracle.stop_loss(threshold)),
            'oracle_contract_quantile_p_star_kw':q_star,
            'oracle_es_p_star_kw':es_star,
        }
        block=df.loc[df.cell_id.eq(cell_id)]
        differences={}
        for column,value in expected_truth.items():
            actual=block[column].to_numpy(dtype=float)
            differences[column+'_max_abs_error']=float(np.max(np.abs(actual-value)))
            if not np.allclose(actual,value,rtol=2e-11,atol=1e-6):
                raise AssertionError(f'raw {column} does not match re-evaluated oracle for {cell_id}; max error={differences[column+"_max_abs_error"]:g}')
        truth_rows.append({'cell_id':cell_id,'p_star':p_star,'oracle_contract_D_kw':contract_d,
                           **expected_truth,**differences})
    pd.DataFrame(truth_rows).to_csv(args.out/'oracle_truth_reconstruction_9_cells.csv',index=False)
    # Retain signed errors on the original scientific scale, alongside the
    # relative-error fields emitted by the base engine. These let the theory-
    # guided audit distinguish the direction of mean/tail-functional shifts.
    df['delta_mean_kw'] = df['estimated_mean_kw'] - df['true_mean_kw']
    df['delta_stoploss_kw'] = df['estimated_stoploss_kw'] - df['true_stoploss_kw']
    df['delta_q99_kw'] = df['estimated_q99_kw'] - df['true_q99_kw']
    df['delta_q_p_star_kw'] = (df['estimated_contract_quantile_p_star_kw']
                              - df['oracle_contract_quantile_p_star_kw'])
    df['delta_es_p_star_kw'] = df['estimated_es_p_star_kw'] - df['oracle_es_p_star_kw']
    cost_cols={'capacity':'estimated_mode_cost_capacity',
               'actual_maximum_demand':'estimated_mode_cost_actual',
               'contracted_maximum_demand':'estimated_mode_cost_contract'}
    df['oracle_value_selected_mode_cny']=[oracle_lookup[c]['oracle_value_'+m+'_cny'] for c,m in zip(df.cell_id,df.selected_mode)]
    df['mode_selection_regret_cny'] = df['oracle_value_selected_mode_cny']-df['oracle_best_value_cny']
    lo_ratio=float(settings['tariff']['delta']); up_ratio=float(settings['tariff']['contract_upper_ratio'])
    within=[]; within_raw=[]; selected_from_regions=[]; geom_ok=[]; pair_cert=[]
    delta_reconstruction_errors=[]; d_mv_reconstructed=[]
    for row in df.itertuples(index=False):
        cell=oracle_by_cell[row.cell_id]
        truecost=cell['oracle']['mode_costs']; vstar=float(cell['oracle']['best_value_cny'])
        lower_d=lo_ratio*float(cell['S_kva']); upper_d=up_ratio*float(cell['S_kva'])
        if not (lower_d-ACTION_FEASIBILITY_TOL_KW <= float(row.estimated_contract_D_kw)
                <= upper_d+ACTION_FEASIBILITY_TOL_KW):
            raise AssertionError(f'infeasible estimated contract action at row {row.replication_id}/{row.method_id}/{row.cell_id}')
        if row.selected_mode==engine.CONTRACT:
            p=cell['policy']; d=float(row.estimated_contract_D_kw)
            action=engine.true_cost_of_estimated_action(engine.CONTRACT,d,oracle,float(cell['S_kva']),p)
            within_loss_raw=float(action-float(truecost[engine.CONTRACT]))
        else: within_loss_raw=0.0
        if within_loss_raw < -REGRET_RECON_TOL_CNY:
            raise AssertionError(f'contract action beats its oracle mode value by {-within_loss_raw:g} CNY at row {row.replication_id}/{row.method_id}/{row.cell_id}')
        within_raw.append(within_loss_raw)
        within.append(max(0.0,within_loss_raw))
        est={m:float(getattr(row,costcol)) for m,costcol in cost_cols.items()}
        if not np.isfinite(list(est.values())).all():
            raise AssertionError(f'nonfinite estimated mode cost at row {row.replication_id}/{row.method_id}/{row.cell_id}')
        delta_map={'capacity':float(row.delta_capacity_cny),
                   'actual_maximum_demand':float(row.delta_actual_cny),
                   'contracted_maximum_demand':float(row.delta_contract_cny)}
        reconstructed_delta={m:est[m]-float(truecost[m]) for m in mode_names}
        delta_error=max(abs(delta_map[m]-reconstructed_delta[m]) for m in mode_names)
        delta_reconstruction_errors.append(delta_error)
        if delta_error>REGRET_RECON_TOL_CNY:
            raise AssertionError(f'signed value errors do not match stored mode costs at row {row.replication_id}/{row.method_id}/{row.cell_id}')
        epsilon=max(abs(v) for v in reconstructed_delta.values())
        if not np.isclose(epsilon,float(row.maximum_mode_value_error_cny),rtol=2e-12,atol=REGRET_RECON_TOL_CNY):
            raise AssertionError('maximum_mode_value_error_cny does not match reconstructed signed errors')
        radius=2*epsilon/float(row.oracle_margin_cny) if float(row.oracle_margin_cny)>0 else np.inf
        d_mv_reconstructed.append(radius)
        if not np.isclose(radius,float(row.d_mv),rtol=2e-12,atol=1e-10):
            raise AssertionError(f'd_MV does not match 2*max|delta|/margin at row {row.replication_id}/{row.method_id}/{row.cell_id}')
        if bool(row.d_mv_below_one)!=(radius<1):
            raise AssertionError('d_mv_below_one does not match reconstructed d_MV')
        min_mode=min(est,key=lambda m:(est[m],m))
        selected_from_regions.append(min_mode); geom_ok.append(min_mode==row.selected_mode)
        mstar=cell['oracle']['selected_mode']
        valid_gaps=[]
        for rival in mode_names:
            if rival==mstar: continue
            gap=float(truecost[rival]-truecost[mstar])
            # map full mode names to extension error fields
            valid_gaps.append(gap > abs(delta_map[mstar])+abs(delta_map[rival])+PAIR_TIE_TOL_CNY)
        pair_cert.append(bool(valid_gaps and all(valid_gaps)))
    df['within_contract_regret_cny']=within
    df['within_contract_regret_raw_cny']=within_raw
    df['mode_region_reconstructed']=selected_from_regions
    df['mode_region_check']=geom_ok
    df['pairwise_gap_certificate']=pair_cert
    df['signed_value_error_reconstruction_max_abs_error_cny']=delta_reconstruction_errors
    df['d_mv_reconstructed']=d_mv_reconstructed
    df['regret_decomposition_error_cny']=df['regret_cny_per_month']-df['mode_selection_regret_cny']-df['within_contract_regret_cny']
    df['regret_decomposition_pass']=df['regret_decomposition_error_cny'].abs()<=REGRET_RECON_TOL_CNY
    contract_quantile_feasible=(df['estimated_contract_quantile_p_star_kw'] >= df['alpha']*lo_ratio*df['S_kva']-1e-7) & (df['estimated_contract_quantile_p_star_kw'] <= df['alpha']*up_ratio*df['S_kva']+1e-7)
    df['unconstrained_es_value_cny']=float(settings['tariff']['demand_rate'])/df['alpha']*df['estimated_es_p_star_kw']
    df['constrained_ru_excess_over_es_cny']=df['estimated_mode_cost_contract']-df['unconstrained_es_value_cny']
    # A single left-inverse p-quantile is only one RU minimizer. With atoms or
    # a CDF plateau, the full minimizer set may intersect the feasible interval
    # even when that selected quantile is outside it. Test equality of optimized
    # values and report point feasibility separately.
    df['estimated_quantile_point_inside_contract_bounds']=contract_quantile_feasible
    df['restricted_ru_equals_unrestricted_es']=(
        df['constrained_ru_excess_over_es_cny'].abs()<=ES_VALUE_EQUALITY_TOL_CNY
    )
    df['es_representation_valid']=df['restricted_ru_equals_unrestricted_es']
    df['es_representation_error_cny']=np.where(
        df['es_representation_valid'],df['constrained_ru_excess_over_es_cny'],np.nan
    )
    if not df['mode_region_check'].all(): raise AssertionError('signed region reconstruction failed')
    selected_cost_from_values=np.asarray([min([getattr(row,c) for c in cost_cols.values()])
                                         for row in df.itertuples(index=False)],dtype=float)
    if not np.allclose(df.estimated_best_value_cny.to_numpy(dtype=float),selected_cost_from_values,
                       rtol=1e-12,atol=REGRET_RECON_TOL_CNY):
        raise AssertionError('estimated_best_value_cny does not match min of stored mode values')
    if not (df.action_correct.to_numpy(dtype=bool)==(df.selected_mode.to_numpy()==df.oracle_mode.to_numpy())).all():
        raise AssertionError('action_correct flag does not match deterministic oracle/estimated modes')
    if not df['regret_decomposition_pass'].all():
        bad=df.loc[~df.regret_decomposition_pass,['method_id','cell_id','regret_decomposition_error_cny']].head()
        raise AssertionError(f'regret decomposition failed: {bad.to_dict(orient="records")}')
    if (df['pairwise_gap_certificate'] & ~df['action_correct']).any(): raise AssertionError('pairwise certificate violation')

    enriched=args.out/'primary_decisions_enriched_10000.csv.gz'
    df.to_csv(enriched,index=False,compression={'method':'gzip','compresslevel':6,'mtime':0})
    method_cell=[]; regret_tails=[]
    tail_rng=np.random.default_rng(20261005)
    for (method,cell),g in df.groupby(['method_id','cell_id'],sort=True):
        k=int(g.action_correct.sum()); n=len(g); low,high=wilson(k,n)
        out={'method_id':method,'cell_id':cell,'oracle_mode':g.oracle_mode.iloc[0],
             'histories':n,'correct_modes':k,'accuracy':k/n,'accuracy_wilson95_low':low,'accuracy_wilson95_high':high,
             'mode_error_count':n-k,'mode_error_frequency':1-k/n}
        for metric in ['regret_cny_per_month','regret_over_oracle_value','mode_selection_regret_cny',
                       'within_contract_regret_cny','delta_mean_kw','delta_stoploss_kw',
                       'delta_q99_kw','delta_q_p_star_kw','delta_es_p_star_kw',
                       'mean_relative_error','stoploss_relative_error',
                       'q99_relative_error','contract_quantile_relative_error_p_star','es_relative_error_p_star',
                       'delta_actual_cny','delta_contract_cny']:
            mean,sd,se,half=mean_ci(g[metric]); out[metric+'_mean']=mean; out[metric+'_sd']=sd
            out[metric+'_mcse']=se; out[metric+'_t95_halfwidth']=half
        out['pairwise_certificate_coverage']=float(g.pairwise_gap_certificate.mean())
        out['d_mv_certificate_coverage']=float((g.d_mv<1).mean())
        method_cell.append(out)
        loss=g.regret_cny_per_month.to_numpy(dtype=float)
        ordered_loss=np.sort(loss)
        qs=np.quantile(loss,[.50,.90,.95,.99,.995,.999,1.0])
        tail={'method_id':method,'cell_id':cell,'oracle_mode':g.oracle_mode.iloc[0],
              'histories':len(loss),'positive_regret_count':int((loss>0).sum()),
              'zero_regret_frequency':float((loss==0).mean()),
              'top_1pct_regret_share':float(ordered_loss[-max(1,int(np.ceil(.01*len(loss)))):].sum()/loss.sum()) if loss.sum()>0 else np.nan,
              'top_0_5pct_regret_share':float(ordered_loss[-max(1,int(np.ceil(.005*len(loss)))):].sum()/loss.sum()) if loss.sum()>0 else np.nan,
              'top_0_1pct_regret_share':float(ordered_loss[-max(1,int(np.ceil(.001*len(loss)))):].sum()/loss.sum()) if loss.sum()>0 else np.nan,
              'max_regret_cny_per_month':float(qs[6]),
              'tail_bootstrap_replicates':int(args.tail_bootstrap),
              'tail_bootstrap_seed':20261005}
        for label,value in zip(['q50','q90','q95','q99','q995','q999'],qs[:-1]):
            tail[label+'_regret_cny_per_month']=float(value)
        for p,label in [(.95,'es95'),(.99,'es99'),(.995,'es995')]:
            tail[label+'_regret_cny_per_month']=empirical_upper_es(loss,p)
        if args.tail_bootstrap>0:
            boot_q=[]
            for start in range(0,args.tail_bootstrap,100):
                b=min(100,args.tail_bootstrap-start)
                sample=loss[tail_rng.integers(0,len(loss),size=(b,len(loss)))]
                boot_q.append(np.quantile(sample,[.95,.99,.995],axis=1).T)
            boot=np.concatenate(boot_q,axis=0)
            for j,label in enumerate(['q95','q99','q995']):
                tail[label+'_bootstrap95_low']=float(np.quantile(boot[:,j],.025))
                tail[label+'_bootstrap95_high']=float(np.quantile(boot[:,j],.975))
        else:
            for label in ['q95','q99','q995']:
                tail[label+'_bootstrap95_low']=np.nan
                tail[label+'_bootstrap95_high']=np.nan
        regret_tails.append(tail)
    pd.DataFrame(method_cell).to_csv(args.out/'method_cell_precision_10000.csv',index=False)
    pd.DataFrame(regret_tails).to_csv(args.out/'regret_tail_summary_by_method_cell_10000.csv',index=False)
    fit_columns=['replication_id','method_id','fit_status','fallback_reason']
    fit_records=df[fit_columns].drop_duplicates()
    if fit_records.duplicated(['replication_id','method_id']).any():
        raise AssertionError('fit status or fallback reason varies across tariff cells within a history-method fit')
    fit_quality=(fit_records.groupby(['method_id','fit_status','fallback_reason'],dropna=False)
                 .size().rename('independent_histories').reset_index())
    fit_quality['share_within_method']=fit_quality['independent_histories']/args.expected_replications
    fit_quality.to_csv(args.out/'fit_status_by_method_10000.csv',index=False)
    contribution=[]
    for (method,oracle_mode),g in df.groupby(['method_id','oracle_mode'],sort=True):
        hist=g.groupby('replication_id',sort=True)[['regret_cny_per_month','mode_selection_regret_cny','within_contract_regret_cny']].mean()
        rec={'method_id':method,'oracle_mode':oracle_mode,'decision_rows':len(g),
             'independent_histories':len(hist),
             'oracle_cells_in_region':int(g.cell_id.nunique())}
        for metric in ['regret_cny_per_month','mode_selection_regret_cny','within_contract_regret_cny']:
            mean,sd,se,half=mean_ci(hist[metric])
            rec[metric+'_mean']=mean; rec[metric+'_sd']=sd; rec[metric+'_mcse']=se
            rec[metric+'_t95_low']=mean-half; rec[metric+'_t95_high']=mean+half
        total=rec['regret_cny_per_month_mean']
        rec['mode_share_of_mean_regret']=rec['mode_selection_regret_cny_mean']/total if total>0 else np.nan
        rec['within_share_of_mean_regret']=rec['within_contract_regret_cny_mean']/total if total>0 else np.nan
        contribution.append(rec)
    pd.DataFrame(contribution).to_csv(args.out/'regret_contributions_by_method_oracle_region_10000.csv',index=False)
    risk_audit=[]
    for method,g in df.groupby('method_id'):
        valid=g[g.es_representation_valid]
        invalid=g[~g.es_representation_valid]
        risk_audit.append({'method_id':method,'rows_total':len(g),
            'rows_restricted_value_equals_unrestricted_es':len(valid),
            'rows_quantile_point_inside_contract_bounds':int(g.estimated_quantile_point_inside_contract_bounds.sum()),
            'rows_restricted_value_above_unrestricted_es':len(invalid),
            'max_abs_error_when_es_representation_valid_cny':float(valid.es_representation_error_cny.abs().max()) if len(valid) else np.nan,
            'minimum_constrained_minus_unrestricted_es_value_cny':float(g.constrained_ru_excess_over_es_cny.min()),
            'max_constrained_minus_unrestricted_es_value_cny':float(g.constrained_ru_excess_over_es_cny.max()),
            'negative_constrained_minus_unrestricted_count':int((g.constrained_ru_excess_over_es_cny < -ES_VALUE_EQUALITY_TOL_CNY).sum()),
            'es_value_equality_tolerance_cny':ES_VALUE_EQUALITY_TOL_CNY})
    pd.DataFrame(risk_audit).to_csv(args.out/'restricted_es_representation_audit.csv',index=False)

    # History-paired method contrasts at both tariff-cell and oracle-region levels.
    contrasts=[]
    for method_a,method_b in itertools.combinations(sorted(df.method_id.unique()),2):
        ga=df[df.method_id==method_a]
        gb=df[df.method_id==method_b]
        joined=ga.merge(gb,on=['replication_id','cell_id','oracle_mode'],suffixes=('_a','_b'),validate='one_to_one')
        for cell,g in joined.groupby('cell_id'):
            x=g['regret_cny_per_month_a']-g['regret_cny_per_month_b']; mean,sd,se,half=mean_ci(x)
            contrasts.append({'scope':'cell','cell_id':cell,'oracle_mode':g.oracle_mode.iloc[0],
                              'method_a':method_a,'method_b':method_b,'histories':len(g),
                              'paired_regret_difference_a_minus_b':mean,'sd':sd,'mcse':se,
                              't95_low':mean-half,'t95_high':mean+half})
        for mode,g0 in joined.groupby('oracle_mode'):
            hist=g0.groupby('replication_id').agg(a=('regret_cny_per_month_a','mean'),b=('regret_cny_per_month_b','mean')).reset_index()
            x=hist.a-hist.b; mean,sd,se,half=mean_ci(x)
            contrasts.append({'scope':'oracle_region','cell_id':'mean_within_region','oracle_mode':mode,
                              'method_a':method_a,'method_b':method_b,'histories':len(hist),
                              'paired_regret_difference_a_minus_b':mean,'sd':sd,'mcse':se,
                              't95_low':mean-half,'t95_high':mean+half})
    pd.DataFrame(contrasts).to_csv(args.out/'paired_regret_contrasts_10000.csv',index=False)

    # Signed estimator-function shifts and region-boundary crossing counts.
    shifts=[]
    for cell,g in df.groupby('cell_id'):
        wide=g.pivot(index='replication_id',columns='method_id',values=['estimated_mean_kw','estimated_stoploss_kw',
                      'estimated_mode_cost_contract','estimated_contract_quantile_p_star_kw','estimated_es_p_star_kw','selected_mode'])
        for left,right in [('KDE','EMP'),('GEV','KDE'),('TAIL','KDE')]:
            rec={'cell_id':cell,'method_a':left,'method_b':right,'histories':len(wide)}
            for var in ['estimated_mean_kw','estimated_stoploss_kw','estimated_mode_cost_contract',
                        'estimated_contract_quantile_p_star_kw','estimated_es_p_star_kw']:
                d=wide[(var,left)]-wide[(var,right)]; m,sd,se,half=mean_ci(d)
                rec[var+'_difference_mean']=m; rec[var+'_difference_mcse']=se; rec[var+'_t95_low']=m-half; rec[var+'_t95_high']=m+half
            rec['selected_mode_switch_count']=int((wide[('selected_mode',left)]!=wide[('selected_mode',right)]).sum())
            rec['selected_mode_switch_frequency']=rec['selected_mode_switch_count']/len(wide)
            shifts.append(rec)
    pd.DataFrame(shifts).to_csv(args.out/'signed_functional_shifts_and_crossings_10000.csv',index=False)
    ke=df[df.method_id.isin(['KDE','EMP'])].pivot(index=['replication_id','cell_id'],columns='method_id',
       values=['selected_mode','estimated_mean_kw','estimated_stoploss_kw','estimated_mode_cost_contract',
               'estimated_contract_quantile_p_star_kw','estimated_es_p_star_kw'])
    order_checks=[]
    for label,variable in [('mean','estimated_mean_kw'),('stop_loss_at_oracle_threshold','estimated_stoploss_kw'),
                           ('optimized_contract_value','estimated_mode_cost_contract'),('ES_p_star','estimated_es_p_star_kw')]:
        d=ke[(variable,'KDE')]-ke[(variable,'EMP')]
        order_checks.append({'claim_tested':f'KDE_minus_EMP_{label}_nonnegative',
            'rows':len(d),'negative_count_beyond_1e-8':int((d < -1e-8).sum()),
            'minimum_difference':float(d.min()),'maximum_difference':float(d.max()),
            'status':'PASS' if (d >= -1e-8).all() else 'COUNTEREXAMPLE'})
    capacity_base=ke[('selected_mode','EMP')]=='capacity'
    capacity_ke=ke[('selected_mode','KDE')]=='capacity'
    cap_bad=capacity_base & ~capacity_ke
    order_checks.append({'claim_tested':'EMP_capacity_implies_KDE_capacity','rows':len(ke),
        'eligible_rows':int(capacity_base.sum()),'violation_count':int(cap_bad.sum()),
        'status':'PASS' if not cap_bad.any() else 'COUNTEREXAMPLE'})
    mode_switch=ke[('selected_mode','KDE')]!=ke[('selected_mode','EMP')]
    order_checks.append({'claim_tested':'global_EMP_to_KDE_mode_invariance',
        'rows':len(ke),'mode_switch_count':int(mode_switch.sum()),
        'mode_switch_frequency':float(mode_switch.mean()),
        'status':'PASS' if not mode_switch.any() else 'COUNTEREXAMPLE'})
    pd.DataFrame(order_checks).to_csv(args.out/'kde_emp_order_falsification_10000.csv',index=False)
    transition=pd.concat([
        ke[('selected_mode','EMP')].rename('EMP_mode'),
        ke[('selected_mode','KDE')].rename('KDE_mode')],axis=1).reset_index()
    transition=transition.merge(df[['replication_id','cell_id','oracle_mode']].drop_duplicates(),
                                on=['replication_id','cell_id'],validate='one_to_one')
    trans_rows=[]
    for (cell,oracle_mode,emp_mode,kde_mode),g in transition.groupby(['cell_id','oracle_mode','EMP_mode','KDE_mode'],sort=True):
        trans_rows.append({'cell_id':cell,'oracle_mode':oracle_mode,'EMP_mode':emp_mode,'KDE_mode':kde_mode,
                           'histories':len(g),'transition_frequency_within_cell':len(g)/args.expected_replications})
    pd.DataFrame(trans_rows).to_csv(args.out/'kde_emp_mode_transition_counts_10000.csv',index=False)
    sw=ke.loc[mode_switch]
    switches=pd.DataFrame({'replication_id':sw.index.get_level_values(0),
        'cell_id':sw.index.get_level_values(1),
        'EMP_mode':sw[('selected_mode','EMP')].to_numpy(),
        'KDE_mode':sw[('selected_mode','KDE')].to_numpy(),
        'delta_mean_kw':(sw[('estimated_mean_kw','KDE')]-sw[('estimated_mean_kw','EMP')]).to_numpy(),
        'delta_stoploss_kw':(sw[('estimated_stoploss_kw','KDE')]-sw[('estimated_stoploss_kw','EMP')]).to_numpy(),
        'delta_contract_value_cny':(sw[('estimated_mode_cost_contract','KDE')]-sw[('estimated_mode_cost_contract','EMP')]).to_numpy(),
        'delta_es_p_star_kw':(sw[('estimated_es_p_star_kw','KDE')]-sw[('estimated_es_p_star_kw','EMP')]).to_numpy()})
    switches.to_csv(args.out/'kde_emp_mode_switch_cases_10000.csv',index=False)

    boundaries=[]; radii=[]
    for c in cells:
        cid=c['cell_id']; truecost=c['oracle']['mode_costs']; v=c['oracle']['selected_mode']
        A=truecost[engine.ACTUAL]; C=truecost[engine.CONTRACT]; K=truecost[engine.CAPACITY]
        boundaries.append({'cell_id':cid,'oracle_mode':v,'oracle_actual_value':A,'oracle_capacity_value':K,
            'oracle_contract_value':C,'boundary_delta_actual_capacity':K-A,
            'boundary_delta_contract_capacity':K-C,
            'boundary_delta_contract_actual_intercept':A-C,
            'estimated_mode_tie_rule':'min(mode_cost, mode_label); equality goes to lexicographically earlier label',
            'equations':'cap=act: delta_actual=K-A; cap=contract: delta_contract=K-C; act=contract: delta_contract=delta_actual+A-C'})
        for target in mode_names:
            if target==v: continue
            radii.append({'cell_id':cid,'oracle_mode':v,'target_mode':target,
                          'infimum_linf_functional_perturbation_to_target_region_cny':l_inf_region_radius(truecost,v,target),
                          'distance_is_to_closed_region':True,
                          'may_not_be_attained_when_target_loses_exact_ties':True})
    pd.DataFrame(boundaries).to_csv(args.out/'decision_region_boundaries_9_cells.csv',index=False)
    pd.DataFrame(radii).to_csv(args.out/'tariff_pair_perturbation_radii.csv',index=False)

    # All pairwise value-order flips relative to their oracle order.
    crossing=[]
    for (method,cell),g in df.groupby(['method_id','cell_id']):
        truecost=oracle_by_cell[cell]['oracle']['mode_costs']
        for m,j in itertools.combinations(mode_names,2):
            oracle_diff=float(truecost[m]-truecost[j])
            est_diff=(g[cost_cols[m]]-g[cost_cols[j]]).to_numpy(dtype=float)
            order_audit=pairwise_order_audit(oracle_diff,est_diff)
            crossing.append({'method_id':method,'cell_id':cell,'mode_a':m,'mode_b':j,
                             'oracle_difference_cny':oracle_diff,
            'oracle_pairwise_near_tie_diagnostic_atol_cny':PAIR_TIE_TOL_CNY,
                             **order_audit})
    pd.DataFrame(crossing).to_csv(args.out/'decision_region_crossings_10000.csv',index=False)

    checks={
        'decision_rows':len(df),'expected_decision_rows':expected,
        'unique_histories':int(df.replication_id.nunique()),
        'unique_keys':not df.duplicated(key).any(),
        'all_mode_costs_finite':bool(np.isfinite(df[list(cost_cols.values())].to_numpy()).all()),
        'all_contract_actions_feasible':bool(((df.estimated_contract_D_kw>=lo_ratio*df.S_kva-ACTION_FEASIBILITY_TOL_KW)&(df.estimated_contract_D_kw<=up_ratio*df.S_kva+ACTION_FEASIBILITY_TOL_KW)).all()),
        'oracle_truth_reconstruction_max_abs_error':float(pd.DataFrame(truth_rows).filter(like='_max_abs_error').to_numpy().max()),
        'signed_region_reconstruction_errors':int((~df.mode_region_check).sum()),
        'signed_value_error_reconstruction_max_abs_error_cny':float(df.signed_value_error_reconstruction_max_abs_error_cny.max()),
        'd_mv_reconstruction_pass':True,
        'regret_decomposition_max_abs_error_cny':float(df.regret_decomposition_error_cny.abs().max()),
        'regret_decomposition_failures':int((~df.regret_decomposition_pass).sum()),
        'within_contract_regret_raw_min_cny':float(df.within_contract_regret_raw_cny.min()),
        'within_contract_regret_raw_below_negative_tolerance_count':int((df.within_contract_regret_raw_cny < -REGRET_RECON_TOL_CNY).sum()),
        'oracle_pair_exact_tie_cell_pair_count':int(sum(float(oracle_by_cell[c]['oracle']['mode_costs'][m]-oracle_by_cell[c]['oracle']['mode_costs'][j]) == 0.0
            for c in oracle_by_cell for m,j in itertools.combinations(mode_names,2))),
        'pairwise_near_tie_diagnostic_atol_cny':PAIR_TIE_TOL_CNY,
        'regret_decomposition_atol_cny':REGRET_RECON_TOL_CNY,
        'restricted_es_representation_max_abs_error_when_valid_cny':float(df.loc[df.es_representation_valid,'es_representation_error_cny'].abs().max()),
        'restricted_es_representation_negative_excess_count':int((df.constrained_ru_excess_over_es_cny < -1e-5).sum()),
        'restricted_es_value_equality_rows':int(df.restricted_ru_equals_unrestricted_es.sum()),
        'restricted_es_quantile_point_feasible_rows':int(df.estimated_quantile_point_inside_contract_bounds.sum()),
        'restricted_es_value_equality_tolerance_cny':ES_VALUE_EQUALITY_TOL_CNY,
        'pairwise_certificate_coverage':float(df.pairwise_gap_certificate.mean()),
        'd_mv_certificate_coverage':float((df.d_mv<1).mean()),
        'pairwise_certificate_violations':int((df.pairwise_gap_certificate & ~df.action_correct).sum()),
        'all_outputs_created':True,
    }
    (args.out/'primary_analysis_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__': main()
