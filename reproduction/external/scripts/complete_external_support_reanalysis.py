"""Small, single-process additions using validated saved results; no refitting.

The external tariffs are hypothetical and fixed before evaluating held-out loads.
S is a training-only reference, not a known factory nameplate capacity. Results
are realized demand-charge losses, never actual bills or distributional regret.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import stats


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def information_pair(a, b):
    maxima = {"K-POINT", "K-EMP", "K-KDE", "K-GEV"}
    detailed = {"K-GTAIL", "K-UGPD", "K-BLOCK"}
    return ("monthly_maxima_same_information" if {a,b} <= maxima else
            "detailed_history_same_information" if {a,b} <= detailed else
            "complete_procedure_different_information")


def paired_stats(d):
    d = np.asarray(d, float)
    assert len(d) == 8 and np.isfinite(d).all()
    avg, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(8))
    radius = float(stats.t.ppf(.975, 7) * se)
    signs = np.array(list(itertools.product((-1., 1.), repeat=8)))
    perm = np.abs((signs * d).mean(axis=1))
    p = float((perm >= abs(avg) - 1e-12*max(1,abs(avg))).mean())
    return {"factories":8, "df":7, "paired_mean_difference":avg,
            "paired_standard_error":se, "paired_t_ci95_low":avg-radius,
            "paired_t_ci95_high":avg+radius,
            "exact_sign_flip_two_sided_p":p, "sign_flip_assignments":256,
            "positive_factory_differences":int((d>0).sum()),
            "negative_factory_differences":int((d<0).sum()),
            "zero_factory_differences":int((d==0).sum())}


def pairwise_tables(factory_values, groupcols, metrics):
    summaries, differences = [], []
    for groups, frame in factory_values.groupby(groupcols, sort=True, dropna=False):
        if not isinstance(groups, tuple): groups=(groups,)
        common=dict(zip(groupcols, groups))
        for metric in metrics:
            pivot=frame.pivot(index="factory", columns="method_id", values=metric)
            assert pivot.shape[0]==8 and pivot.notna().all().all()
            for a,b in itertools.combinations(sorted(pivot.columns),2):
                d=pivot[a]-pivot[b]
                c={**common,"metric":metric,"method_a":a,"method_b":b,
                   "contrast":a+" minus "+b,"information_comparison":information_pair(a,b),
                   "inference_scope":"descriptive sensitivity; sign symmetry is an additional assumption"}
                summaries.append({**c,**paired_stats(d.to_numpy())})
                differences.extend({**c,"factory":f,"paired_difference":float(x)} for f,x in d.items())
    return pd.DataFrame(summaries),pd.DataFrame(differences)


def external(root, out):
    src=root/"data/local_full_20261001/summary/external"
    valid=json.loads((src/"final_external_validation.json").read_text(encoding="utf-8"))
    assert valid["status"]=="PASS"
    scores=pd.read_csv(src/"all_proper_scores.csv.gz")
    metrics=["crps","qwcrps_090","q95_pinball","q99_pinball","q95_exceedance_brier","q99_exceedance_brier"]
    assert len(scores)==448 and not scores.duplicated(["variant","factory","origin_id","method_id"]).any()
    assert (scores.groupby(["variant","factory","method_id"]).size()==4).all()
    fs=scores.groupby(["variant","factory","method_id"],as_index=False)[metrics].mean()
    t,d=pairwise_tables(fs,["variant"],metrics)
    t.to_csv(out/"external_score_paired_t_exact_signflip.csv",index=False)
    d.to_csv(out/"external_score_factory_paired_differences.csv",index=False)
    cells=pd.read_csv(root/"data/primary_no_upper_20261001/summary/primary/oracle_mode_values_9_cells.csv")[["cell_id","alpha","utilization"]]
    fixed=cells.assign(rd=36.,rc=22.5,kappa=2.,delta=.4)
    fixed.to_csv(out/"external_representative_tariff_design.csv",index=False)
    rows=[]
    for variant in sorted(scores.variant.unique()):
        vs=src/"variants"/variant
        index=pd.read_csv(vs/"predictive_draw_index.csv")
        with np.load(vs/"predictive_draws_normalized.npz",allow_pickle=False) as archive:
            for record in index.to_dict("records"):
                x=np.asarray(archive[record["raw_draw_key"]],float)
                assert len(x)==50000 and np.isfinite(x).all() and (x>=0).all()
                s=scores[(scores.variant==variant)&(scores.factory==record["factory"])&
                         (scores.origin_id==record["origin_id"])&(scores.method_id==record["method_id"])]
                assert len(s)==1
                sr=s.iloc[0]
                assert np.isclose(float(sr.normalization_kw),float(record["normalization_kw"]),rtol=1e-13)
                # All forecast quantities are computed before reading y.
                x=np.sort(x)
                suffix=np.r_[np.cumsum(x[::-1])[::-1],0.]
                mu=float(x.mean())
                for cell in fixed.to_dict("records"):
                    alpha,u,kappa=cell["alpha"],cell["utilization"],cell["kappa"]
                    p=1-1/(kappa*alpha)
                    q=float(x[max(0,int(np.ceil(p*len(x)))-1)])
                    S=1/u
                    D=float(np.clip(q/alpha,.4*S,S))
                    z=alpha*D
                    j=int(np.searchsorted(x,z,side="right"))
                    H=float((suffix[j]-(len(x)-j)*z)/len(x))
                    costs={"actual_maximum_demand":36*mu,"capacity":22.5*S,
                           "contracted_maximum_demand":36*(D+kappa*H)}
                    selected=min(costs,key=lambda m:(costs[m],m))
                    # Held-out y appears only in evaluation, never action choice.
                    y=float(sr.observed_normalized_max)
                    realized={"actual_maximum_demand":36*y,"capacity":22.5*S,
                              "contracted_maximum_demand":36*(D+kappa*max(y-z,0))}[selected]
                    rows.append({"variant":variant,"factory":record["factory"],
                        "origin_id":record["origin_id"],"test_month":record["test_month"],
                        "method_id":record["method_id"],"fit_status":record["fit_status"],
                        **cell,"p_star":p,"training_reference_kw":float(record["normalization_kw"]),
                        "hypothetical_S_kw":S*float(record["normalization_kw"]),
                        "estimated_D_kw":D*float(record["normalization_kw"]),
                        "forecast_mean_normalized":mu,"forecast_q_pstar_normalized":q,
                        "forecast_stoploss_at_selected_D_normalized":H,
                        "estimated_best_charge_per_training_reference_kw":costs[selected],
                        "selected_mode":selected,
                        "realized_charge_per_training_reference_kw":realized,
                        "realized_hypothetical_charge_cny":realized*float(record["normalization_kw"]),
                        "observed_test_max_kw":y*float(record["normalization_kw"])})
    r=pd.DataFrame(rows)
    assert len(r)==4032 and not r.duplicated(["variant","factory","origin_id","method_id","cell_id"]).any()
    assert np.isfinite(r.select_dtypes("number").to_numpy()).all()
    assert (r.estimated_D_kw>=.4*r.hypothetical_S_kw-1e-8).all() and (r.estimated_D_kw<=r.hypothetical_S_kw+1e-8).all()
    r.to_csv(out/"external_representative_tariff_realized_rows.csv",index=False)
    m=["realized_charge_per_training_reference_kw","realized_hypothetical_charge_cny"]
    f=r.groupby(["variant","cell_id","alpha","utilization","factory","method_id"],as_index=False)[m].mean()
    f.to_csv(out/"external_realized_factory_values.csv",index=False)
    t,d=pairwise_tables(f,["variant","cell_id","alpha","utilization"],m)
    t.to_csv(out/"external_realized_paired_t_exact_signflip.csv",index=False)
    d.to_csv(out/"external_realized_factory_paired_differences.csv",index=False)
    r.groupby(["variant","cell_id","method_id","selected_mode"],as_index=False).size().to_csv(out/"external_realized_mode_counts.csv",index=False)
    r.groupby(["variant","cell_id","method_id"],as_index=False)[m].mean().to_csv(out/"external_realized_method_cell_summary.csv",index=False)
    return {"status":"PASS","prediction_units":64,"methods":7,"tariff_cells":9,"decisions":len(r),
            "original_draws_modified":False,"refitted_distributions":False,
            "target":"observed next-month hypothetical demand charge, not tariff bills or oracle regret",
            "reference_rule":"S = mean(training monthly maxima) / prespecified utilization; no future input",
            "small_sample_inference":"t df=7 and all 256 paired sign flips; descriptive, no confirmatory multiplicity claim"}


def support(root,out):
    src=root/"data/primary_no_upper_20261001/support_policy_audit"
    manifest=json.loads((src/"first1000_training_legacy.audit-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"]=="PASS" and manifest["range"]==[0,1000]
    cols=["replication_id","support_policy","method_id","cell_id","oracle_mode","alpha",
          "utilization","estimated_mean_kw","estimated_stoploss_kw","regret_cny_per_month",
          "selected_mode","action_correct","fit_status","delta_actual_cny","delta_contract_cny"]
    frames=[]
    for run,folder in (("sensitivity",src/manifest["signature"][:16]),("no_upper",src/"de922a595401759a")):
        for start in range(0,1000,50):
            p=folder/f"rep_{start:05d}_{start+49:05d}.csv"
            meta=json.loads(p.with_name(p.stem+"_meta.json").read_text(encoding="utf-8"))
            assert meta["status"]=="PASS" and sha(p)==meta["csv_sha256"]
            frames.append(pd.read_csv(p,usecols=cols))
    f=pd.concat(frames,ignore_index=True)
    assert len(f)==225000 and f.support_policy.nunique()==5
    assert not f.duplicated(["replication_id","support_policy","method_id","cell_id"]).any()
    assert np.isfinite(f.select_dtypes("number").to_numpy()).all()
    groups=["support_policy","method_id","cell_id","oracle_mode","alpha","utilization"]
    means=f.groupby(groups,as_index=False).agg(histories=("replication_id","size"),
         mean_estimated_mean_kw=("estimated_mean_kw","mean"),
         mean_estimated_stoploss_kw=("estimated_stoploss_kw","mean"),
         mean_regret_cny=("regret_cny_per_month","mean"),mode_accuracy=("action_correct","mean"))
    means.to_csv(out/"support_five_policy_levels_first1000.csv",index=False)
    keys=["replication_id","method_id","cell_id","oracle_mode","alpha","utilization"]
    metrics=["estimated_mean_kw","estimated_stoploss_kw","regret_cny_per_month","delta_actual_cny","delta_contract_cny"]
    base=f[f.support_policy=="no_upper"].set_index(keys)
    summaries=[]
    for policy in sorted(set(f.support_policy)-{"no_upper"}):
        cmp=f[f.support_policy==policy].set_index(keys)
        assert cmp.index.equals(base.index) or set(cmp.index)==set(base.index)
        cmp=cmp.reindex(base.index)
        diff=cmp[metrics]-base[metrics]
        diff["mode_changed"]=(cmp.selected_mode!=base.selected_mode).astype(float)
        diff["accuracy_difference"]=cmp.action_correct.astype(float)-base.action_correct.astype(float)
        diff=diff.reset_index()
        for key,g in diff.groupby(keys[1:]):
            row={"comparison_policy":policy,"reference_policy":"no_upper",**dict(zip(keys[1:],key)),"paired_histories":len(g)}
            for metric in metrics+["mode_changed","accuracy_difference"]:
                d=g[metric].to_numpy(float)
                row[metric+"_paired_mean_difference"]=float(d.mean())
                row[metric+"_paired_se"]=float(d.std(ddof=1)/np.sqrt(len(d)))
                row[metric+"_maximum_abs_difference"]=float(np.max(np.abs(d)))
            summaries.append(row)
    s=pd.DataFrame(summaries)
    s.to_csv(out/"support_five_policy_paired_vs_no_upper_first1000.csv",index=False)
    ranks=means.copy()
    ranks["mean_regret_rank_within_policy_cell"]=ranks.groupby(["support_policy","cell_id"]).mean_regret_cny.rank(method="min")
    ranks.to_csv(out/"support_five_policy_method_rank_first1000.csv",index=False)
    unchanged=s[s.method_id.isin(["EMP","KDE","EVENT-EMP"])]
    assert (unchanged["regret_cny_per_month_maximum_abs_difference"]<=1e-7).all()
    assert (unchanged["estimated_mean_kw_maximum_abs_difference"]<=1e-7).all()
    assert (unchanged["estimated_stoploss_kw_maximum_abs_difference"]<=1e-7).all()
    return {"status":"PASS","paired_histories":1000,"policies":sorted(f.support_policy.unique()),
            "decision_rows":len(f),"same_keys_all_policies":True,
            "EMP_KDE_EVENT_EMP_unchanged":True,
            "scope":"first 1000 paired histories; legacy oracle bound is diagnostic, not valid deployment input"}


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args()
    out=args.root/"data/final_requirements_audit_20261002";out.mkdir(parents=True,exist_ok=True)
    t=time.time()
    result={"external":external(args.root,out),"support":support(args.root,out)}
    result["elapsed_seconds"]=time.time()-t
    result["source_script_sha256"]=sha(__file__)
    result["output_sha256"]={str(x.name):sha(x) for x in sorted(out.glob("*.csv"))}
    write_json(out/"external_support_additions_checks.json",result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()
