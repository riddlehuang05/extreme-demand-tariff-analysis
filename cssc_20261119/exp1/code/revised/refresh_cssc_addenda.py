"""Integrate verified CSSC review experiments."""
from pathlib import Path
import hashlib, json, shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[3]
V=R/"CSSC_REVIEW_REVISION_20261112"
W=V/"adaptive_w1_20261119"/"combined_full"
A=V/"tail_ablation_20261119"
S=R/"CSSC_SUBMISSION_20261112"/"Supporting_Information"
M=R/"CSSC_SUBMISSION_20261112"/"Manuscript"/"Manuscript.tex"
D=V/"source_data"/"data"
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def once(t,a,b):
    if t.count(a)!=1: raise ValueError((a[:70],t.count(a)))
    return t.replace(a,b,1)
def main():
    manifest=json.loads((W/"adaptive_w1_output_manifest.json").read_text())
    assert manifest["status"]=="PASS"
    for n,h in manifest["outputs_sha256"].items(): assert sha(W/n)==h,n
    run=json.loads((A/"run_manifest.json").read_text())
    assert run["status"]=="PASS" and run["decision_rows"]==540000
    methods=("TAIL","EVENT-EMP","GEV","KDE","EMP")
    w=pd.read_csv(W/"adaptive_w1_by_method.csv").query("bound == 'combined'").set_index("method_id")
    c=pd.read_csv(W/"adaptive_w1_coverage.csv").set_index("condition")
    labels=(("Primitive numerical check","primitive"),("Contrast-Wasserstein numerical check","combined_direct_pair_contrast"),("Mode-wise pairwise-Wasserstein numerical check","combined_modewise_per_competitor"),("Uniform-Wasserstein numerical check","combined_uniform_L_full_action"),("Distributional envelope","coarse_distributional"))
    src=D/"derived_publication"/"tariff_contrast_wasserstein"
    f=pd.read_csv(src/"FigureS3_W1_input.csv")
    for i,row in f.iterrows():
        q=w.loc[row.method_id]
        f.loc[i,["w1_mean_kw","cdf_lower_mean_kw","cdf_upper_mean_kw","cdf_interval_width_mean_kw"]]=[q.bracket_midpoint_mean_kw,q.lower_mean_kw,q.upper_mean_kw,q.width_mean_kw]
    f.to_csv(src/"FigureS3_W1_input.csv",index=False)
    cov=pd.DataFrame([dict(condition=label,total_rows=450000,passing_rows=int(c.loc[key].passing_rows),coverage_percent=float(c.loc[key].coverage_percent),observed_mode_selection_violations=0) for label,key in labels])
    for n in ("FigureS3_coverage_input.csv","numerical_condition_coverage.csv"):cov.to_csv(src/n,index=False)
    old=pd.read_csv(src/"FigureS3_primitive_cell_input.csv")
    cells=old[["cell_id","policy_id"]].drop_duplicates().sort_values(["policy_id","cell_id"])
    colors=dict(zip(methods,("#176B87","#985D88","#D17828","#39836C","#666666")))
    marks=dict(zip(methods,("o","s","^","D","v")))
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":8,"pdf.fonttype":42})
    fig,ax=plt.subplots(1,3,figsize=(8.2,3.1))
    fig.subplots_adjust(left=.08,right=.98,bottom=.29,top=.86,wspace=.40)
    for i,m in enumerate(methods):
        q=w.loc[m]; y=4-i
        ax[0].plot([q.lower_mean_kw/1000,q.upper_mean_kw/1000],[y,y],color=colors[m])
        ax[0].plot(q.bracket_midpoint_mean_kw/1000,y,marker=marks[m],color=colors[m],markersize=4)
        p=old.loc[old.method_id.eq(m)].set_index("cell_id").loc[cells.cell_id]
        ax[1].plot(range(9),100*p.numerical_certificate_coverage.to_numpy(float),linestyle="none",marker=marks[m],color=colors[m],markersize=3.1)
    ax[0].set_yticks(range(5),methods[::-1]); ax[0].set_xlabel("Mean $W_1$ (MW)"); ax[0].set_title("(a)  Numerical $W_1$ brackets",loc="left")
    ax[1].set_xticks(range(9),[str(i) for i in range(1,10)])
    ax[1].set_ylabel("Coverage (%)"); ax[1].set_title("(b)  Primitive check",loc="left")
    vals=cov.coverage_percent.to_numpy(float)
    ax[2].barh(range(5),vals,color="#777777",height=.58)
    ax[2].set_yticks(range(5),["Primitive","Contrast $W_1$","Mode-wise $W_1$","Full-action $W_1$","Envelope"],fontsize=7)
    ax[2].invert_yaxis(); ax[2].set_xlim(0,1.25*max(vals)); ax[2].set_xlabel("Decision coverage (%)"); ax[2].set_title("(c)  Numerical checks",loc="left")
    for i,v in enumerate(vals):ax[2].text(v+.08,i,f"{v:.2f}%",va="center",fontsize=7)
    for a in ax:a.grid(axis="y",color="#dddddd",linewidth=.4); a.set_axisbelow(True)
    for ext in ("pdf","png","svg"):fig.savefig(src/f"FigureS3.{ext}",dpi=400 if ext=="png" else None)
    plt.close(fig); shutil.copy2(src/"FigureS3.pdf",S/"figures"/"FigureS3.pdf")
    audit=json.loads((src/"audit_summary.json").read_text()); audit["coverage"]=cov.to_dict("records"); audit["figure_pdf_sha256"]=sha(src/"FigureS3.pdf"); audit["adaptive_w1_manifest_sha256"]=sha(W/"adaptive_w1_output_manifest.json")
    (src/"audit_summary.json").write_text(json.dumps(audit,indent=2))
    p=S/"tables"/"Table_S6.tex"; t=p.read_text()
    for m in sorted(methods):
        o=next(x for x in t.splitlines() if x.startswith(m+" & "))
        q=w.loc[m]; t=t.replace(o,f"{m} & {q.bracket_midpoint_mean_kw/1000:.3f} & {q.lower_mean_kw/1000:.3f} & {q.upper_mean_kw/1000:.3f}"+r"\\",1)
    i=t.index(r"\noindent\footnotesize"); j=t.index(r"\endgroup",i)
    t=t[:i]+r"\noindent\footnotesize Point values are midpoints of adaptive CDF brackets. Columns average 10,000 fitted laws per method. Bounds include an analytic tail allowance; they are numerical integration bounds."+"\n"+t[j:];p.write_text(t)
    p=S/"tables"/"Table_S14.tex";t=p.read_text()
    for label,key in labels[:4]:
        o=next(x for x in t.splitlines() if x.startswith(label+" & "))
        q=c.loc[key];t=t.replace(o,f"{label} & {int(q.passing_rows):,} & {q.coverage_percent:.6f} & 0 "+r"\\",1)
    t=t.replace(r"$2LW_{1,\rm upper}+\zeta_{\rm full,num}<\gamma_F$",r"$LW_{1,\rm upper}+\zeta_{\rm full,num}<\gamma_F$")
    p.write_text(t)
    out=D/"source"/"adaptive_w1";out.mkdir(parents=True,exist_ok=True)
    for n in (*manifest["outputs_sha256"],"adaptive_w1_output_manifest.json"):shutil.copy2(W/n,out/n)
    out=D/"source"/"tail_ablation";out.mkdir(parents=True,exist_ok=True)
    for n in ("history_cell_metrics.parquet","history_fit_checks.parquet","paired_summary.csv","run_manifest.json"):shutil.copy2(A/n,out/n)
    common=V/"high_pstar_cv_crps_20261115";contract=V/"contract_action_errors_20261119"
    for folder,name in ((common,"high_pstar_common_contract_cells.csv"),(contract,"contract_action_error_by_method_cell.csv")):
        out=D/"source"/folder.name;out.mkdir(parents=True,exist_ok=True);shutil.copy2(folder/name,out/name)
        for p in folder.glob("*audit*.json"):shutil.copy2(p,out/p.name)
    update_text(A,common)
    print("Integrated CSSC addenda")
def update_text(A,common):
    m=M.read_text();i=m.index("Using the same numerically evaluated upper bounds on");j=m.index("Mode preservation does not by itself",i)
    m=m[:i]+r"""Using adaptive numerical $W_1$ upper bounds and the same solver
allowance, the contrast-Wasserstein check passed for $59{,}410/450{,}000$
decisions ($13.202\%$), versus $27{,}352$ ($6.078\%$) for the
mode-wise pairwise check. The sharp full-action check
$LW_{1,\rm upper}+\zeta_{\rm full,num}<\gamma_F$ passed for
$31{,}677$ ($7.039\%$). No observed mode-selection errors occurred
among passing rows. These numerical check pass rates use the known
generating law in simulation (Table~S14; Figure~S3).

"""+m[j:]
    m=once(m,"The complete-procedure TAIL--KDE comparison uses different information",r"""A matched event-level comparison holds TAIL's selected threshold,
count law, and subthreshold component fixed while replacing its GPD
excess distribution with the empirical excess law. Across 10,000
histories per setting, TAIL lowered nine-cell mean regret by $1.911$
thousand CNY/month (95\% paired interval $[1.681,2.141]$) under
Poisson--GPD and by $2.060$ ($[1.807,2.313]$) under
negative-binomial--Weibull. Regional results are in Table~S18.

The complete-procedure TAIL--KDE comparison uses different information""")
    M.write_text(m)
    p=S/"Supporting_Information.tex";s=p.read_text();i=s.index("The numerical Wasserstein upper quantity is based on CDF bounds;");j=s.index(r"\subsection*{S1.8",i)
    s=s[:i]+r"""The adaptive numerical Wasserstein bracket integrates the CDF
difference on a nonuniform log-demand grid, refines the intervals with
the largest uncertainty, and adds an analytic right-tail allowance.
The full-action regret comparison permits the sharper uniform check
\[LW_{1,\rm upper}+\zeta_{\rm full,num}<\gamma_F.\]
The preceding $2L$ condition remains a valid mode-value comparator;
Table~\ref{tab:numerical-sufficient-condition-audit} reports the
sharper full-action check. Across 50,000 fitted laws, 49,936 adaptive
brackets had width at most 300 kW. The GEV median bracket width was
260.972 kW, its 95th percentile was 296.453 kW, and its 99th
percentile was 299.634 kW. Table~S6 reports
method-level brackets and Figure~S3 shows the revised checks.
The primitive check passed in $24{,}894/450{,}000$ decisions
($5.532\%$), the direct contrast check in $59{,}410$ ($13.202\%$),
the per-competitor mode-wise check in $27{,}352$ ($6.078\%$), and the
sharp full-action check in $31{,}677$ ($7.039\%$). No observed mode
errors occurred among passing rows. The known generating law supplies
these simulation diagnostics.

"""+s[j:]
    figpath=S/"FIGURES.tex"; figs=figpath.read_text()
    figs=once(figs,"logarithmic axis retains the loose GEV upper bound.","bounds use adaptive CDF integration and an analytic tail allowance.")
    figs=once(figs,"Mean $W_1$ point estimates and CDF-based numerical","Mean bracket midpoints and adaptive numerical")
    figs=once(figs,"uniform-Wasserstein, and coarse distributional-envelope checks over","sharp full-action Wasserstein, and coarse distributional-envelope checks over")
    figpath.write_text(figs)
    s=once(s,r"\input{tables/Table_S14.tex}"+"\n"+r"\FloatBarrier",r"\input{tables/Table_S14.tex}"+"\n"+r"\FloatBarrier"+"\n"+r"\input{tables/Table_S17.tex}"+"\n"+r"\FloatBarrier"+"\n"+r"\input{tables/Table_S18.tex}"+"\n"+r"\FloatBarrier"+"\n"+r"\input{tables/Table_S19.tex}"+"\n"+r"\FloatBarrier")
    s=once(s,r"\input{tables/Table_S17.tex}",r"""In the primary contract-optimal cells, the tariff-implied quantile
level is $p^*=0.5833$. TAIL's mean absolute $q_{p^*}$ error is
$2.695\%$ of the oracle quantile, versus $4.088\%$ for KDE. At the
interior clear-contract cell, mean $|\widehat D-D_F^*|/S$ is
$1.385\%$ for TAIL and $2.100\%$ for KDE; at the lower-bound cell,
it is $0.0127\%$ and $0.1818\%$. Cell-level errors and their
correlations with regret are in the source data.

\input{tables/Table_S17.tex}""")
    p.write_text(s)
    checks=pd.read_parquet(W/"adaptive_w1_decision_checks.parquet")
    t=r"""\begin{table}[tbp]\centering\small
\caption{Adaptive Wasserstein numerical checks by method.}\label{tab:adaptive-w1-method}
\begin{tabular}{@{}lrrr@{}}\toprule Method & Contrast (\%) & Mode-wise (\%) & Full-action (\%)\\\midrule
"""
    for method in ("TAIL","EVENT-EMP","GEV","KDE","EMP"):
        q=checks.loc[checks.method_id.eq(method)]
        t+=f"{method} & {100*q.combined_direct_pair_contrast.mean():.2f} & {100*q.combined_modewise_per_competitor.mean():.2f} & {100*q.combined_uniform_L_full_action.mean():.2f}"+r"\\"+"\n"
    t+=r"\bottomrule\end{tabular}\par\smallskip\footnotesize Each percentage uses 90,000 method-specific decisions. The known generating law supplies oracle gaps and Wasserstein values.\end{table}"
    (S/"tables"/"Table_S17.tex").write_text(t)
    a=pd.read_csv(A/"paired_summary.csv")
    t=r"""\begin{table}[tbp]\centering\small
\caption{Matched event-level excess-law ablation: TAIL minus empirical excess.}\label{tab:tail-excess-ablation}
\begin{tabular}{@{}llrr@{}}\toprule Setting & Region & Mean & 95\% paired interval\\\midrule
"""
    for dgp in ("primary","nb_weibull"):
        for region in ("actual","contract","capacity","all_nine_cells"):
            q=a.loc[(a.dgp==dgp)&(a.oracle_region==region)&(a.alternative=="TAIL-EMP-EXCESS")&(a.metric=="regret_cny")].iloc[0]
            t+=f"{('Poisson--GPD' if dgp=='primary' else 'NB--Weibull')} & {region.replace('_',' ')} & {q.mean_diff_full_minus_alternative/1000:+.3f} & [{q.ci95_lower/1000:+.3f}, {q.ci95_upper/1000:+.3f}]"+r"\\"+"\n"
    t+=r"\bottomrule\end{tabular}\par\smallskip\footnotesize Units are thousand CNY/month. Each setting uses 10,000 paired 24-month histories. The comparator retains TAIL's selected threshold, count law, and subthreshold component, replacing the excess CDF. Negative differences favour TAIL. The fixed-0.95 sensitivity gives nine-cell differences of $-0.088$ [$-0.265,0.089$] and $-0.060$ [$-0.260,0.140$] in the two settings; for histories selected at another threshold, that sensitivity omits the 500-resample shape-width gate.\end{table}"
    (S/"tables"/"Table_S18.tex").write_text(t)
    x=pd.read_csv(common/"high_pstar_common_contract_cells.csv")
    t=r"""\begin{table}[tbp]\centering\small
\caption{High-$p^*$ contrasts on four common contract-optimal tariff cells.}\label{tab:common-high-pstar}
\begin{tabular}{@{}rlrrr@{}}\toprule $p^*$ & Contrast & Mean & 95\% paired interval & Normalized (pp)\\\midrule
"""
    for q in x.itertuples(index=False):
        t+=f"{q.p_star:.2f} & {q.method_a}--{q.method_b} & {q.difference_cny_mean/1000:+.3f} & [{q.difference_cny_95ci_low/1000:+.3f}, {q.difference_cny_95ci_high/1000:+.3f}] & {q.normalized_difference_percentage_points_mean:+.3f}"+r"\\"+"\n"
    t+=r"\bottomrule\end{tabular}\par\smallskip\footnotesize Regret units are thousand CNY/month; the common utilization cells are 0.43, 0.45, 0.50, and 0.625. Each row uses 10,000 paired histories. The penalty coefficient changes with $p^*$, so the rows compare joint fee configurations.\end{table}"
    (S/"tables"/"Table_S19.tex").write_text(t)
    p=S/"README.txt";p.write_text(p.read_text().replace("S1-S16","S1-S19").replace("S14-S16","S14-S19"))
if __name__=="__main__":main()

