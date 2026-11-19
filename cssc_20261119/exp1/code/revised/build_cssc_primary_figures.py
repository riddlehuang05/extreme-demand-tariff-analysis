"""Rebuild CSSC main Figures 1 and 3 from the revised primary summaries.

This is deterministic plotting from verified analysis outputs; no fits or
Monte Carlo histories are generated here.
"""
from pathlib import Path
import json, shutil, hashlib, string, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
WORKSPACE=Path(__file__).resolve().parents[3]
ROOT=WORKSPACE/'CSSC_REVIEW_REVISION_20261112'/'source_data'
SRC=ROOT/'data/source'; PLOT=ROOT/'data/plot'; PLOT.mkdir(parents=True,exist_ok=True)
(ROOT/'audit').mkdir(parents=True,exist_ok=True)
(WORKSPACE/'CSSC_SUBMISSION_20261112'/'Manuscript'/'figures').mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.4,'ytick.labelsize':7.4,'legend.fontsize':7.2,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'lines.linewidth':1.15,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','savefig.facecolor':'white','figure.facecolor':'white','mathtext.fontset':'dejavuserif','axes.unicode_minus':True})
M=['TAIL','EVENT-EMP','GEV','KDE','EMP']; C=dict(zip(M,['#176B87','#985D88','#D17828','#39836C','#666666'])); MK=dict(zip(M,['o','s','^','D','v']))
REG=['actual_maximum_demand','contracted_maximum_demand','capacity']; RL=dict(zip(REG,['Actual','Contract','Capacity']))
AUD=[]; MAN=[]
def read(name):
 p=SRC/name
 assert p.is_file(), p
 d=pd.read_csv(p);return d

def js(name):
 p=SRC/name
 assert p.is_file(), p
 return json.loads(p.read_text())
def plotdata(name,d):d.to_csv(PLOT/(name+'.csv'),index=False)
def axstyle(ax,title=None):
 ax.grid(axis='y',color='#dddddd',linewidth=.4);ax.set_axisbelow(True)
 if title:ax.set_title(title,loc='left',pad=7,fontweight='bold')
def panel(ax,letter,title):axstyle(ax,f'({letter})  {title}')
def legend_methods(fig,y=1.0,methods=M):
 h=[Line2D([],[],color=C[m],marker=MK[m],lw=1,ms=4,label=m) for m in methods]
 fig.legend(handles=h,loc='upper center',bbox_to_anchor=(.52,y),ncol=len(methods),frameon=False,columnspacing=1.5)
def save(fig,name,section):
 if section!='main' or name not in ('Figure1','Figure3'):
  raise ValueError('This figure builder updates only CSSC main Figures 1 and 3')
 path=WORKSPACE/'CSSC_SUBMISSION_20261112'/'Manuscript'/'figures'/name.replace('Figure','Figure_')
 fig.canvas.draw()
 from matplotlib.text import Text
 renderer=fig.canvas.get_renderer();fw,fh=fig.canvas.get_width_height()
 outside=[]
 dormant=set()
 for axis in fig.axes:
  for ticks,lim in [(axis.xaxis.get_major_ticks()+axis.xaxis.get_minor_ticks(),axis.get_xlim()),(axis.yaxis.get_major_ticks()+axis.yaxis.get_minor_ticks(),axis.get_ylim())]:
   low,high=sorted(lim)
   for tick in ticks:
    if not low-1e-12 <= tick.get_loc() <= high+1e-12:dormant.update([tick.label1,tick.label2])
 for tx in fig.findobj(Text):
  if tx in dormant or not tx.get_visible() or not tx.get_text():continue
  box=tx.get_window_extent(renderer)
  if box.x0 < -1 or box.y0 < -1 or box.x1 > fw+1 or box.y1 > fh+1:outside.append(tx.get_text())
 AUD.append({'check':name+' text within canvas','passed':not outside,'outside':outside})
 from io import BytesIO
 import os
 for ext in ['pdf','svg','eps','png']:
  buf=BytesIO();fig.savefig(buf,format=ext,dpi=600 if ext=='png' else 160)
  payload=buf.getvalue();tmp=path.with_suffix('.tmp.'+ext)
  with tmp.open('wb') as out:out.write(payload);out.flush();os.fsync(out.fileno())
  tmp.replace(path.with_suffix('.'+ext))
  assert path.with_suffix('.'+ext).read_bytes()==payload
 with Image.open(path.with_suffix('.png')) as im:
  buf=BytesIO();im.convert('RGB').save(buf,format='TIFF',compression='tiff_lzw',dpi=(600,600))
  path.with_suffix('.tiff').write_bytes(buf.getvalue())
 buf=BytesIO();fig.savefig(buf,format='png',dpi=160);(ROOT/'audit'/(name+'_preview.png')).write_bytes(buf.getvalue())
 MAN.append({'name':name,'section':section,'width_mm':fig.get_figwidth()*25.4,'height_mm':fig.get_figheight()*25.4,'formats':['pdf','svg','eps','png','tiff']})
 plt.close(fig)
def interval(ax,y,v,l,h,col='black',marker='o',ms=4):
 ax.errorbar(v,y,xerr=[[v-l],[h-v]],fmt=marker,color=col,ms=ms,elinewidth=1.0,capsize=2,mew=.7,zorder=4)
def pickpair(d,a,b):
 g=d[(d.method_a==a)&(d.method_b==b)]
 if len(g):return g,1
 g=d[(d.method_a==b)&(d.method_b==a)];assert len(g);return g,-1

def oriented(d,a,b,value,lo,hi):
 g,s=pickpair(d,a,b);assert len(g)==1
 r=g.iloc[0];v=s*r[value];l=r[lo] if s==1 else -r[hi];h=r[hi] if s==1 else -r[lo];assert l<=v<=h
 return v,l,h
p=read('primary/method_cell_precision_10000.csv')
o=read('primary/oracle_mode_values_9_cells.csv').sort_values(['alpha','utilization'])
CELLS=list(o.cell_id); O=o.set_index('cell_id'); short={r.cell_id:f'{r.utilization:.3f}' for r in o.itertuples()}
cell_labels=[f'{r.alpha:.2f}/{r.utilization:.3f}' for r in o.itertuples()]
assert len(p)==45 and p.histories.eq(10000).all(); assert p.groupby('method_id').size().eq(9).all()
assert np.allclose(p.regret_cny_per_month_mean,p.mode_selection_regret_cny_mean+p.within_contract_regret_cny_mean,atol=1e-6)
AUD.append({'check':'primary 45 summaries, 10,000 histories; loss decomposition','passed':True})

# Category positions are equally spaced discrete cells, grouped by alpha.
def group_cells(ax,design):
 n=len(design);ax.set_xticks(range(n),[f'{u:.3f}'.lstrip('0') for u in design.utilization],rotation=0)
 ax.set_xlabel('Utilization $u$',labelpad=25)
 for alpha,grp in design.reset_index(drop=True).groupby('alpha',sort=False):
  lo,hi=grp.index.min(),grp.index.max()
  ax.plot([lo-.35,hi+.35],[-.145,-.145],transform=ax.get_xaxis_transform(),color='.45',lw=.55,clip_on=False)
  ax.text((lo+hi)/2,-.185,rf'$\alpha={alpha:.2f}$',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=7)
  if lo:ax.axvline(lo-.5,color='.65',ls=':',lw=.65)
# Figure 1: preserve validated deterministic geometry, upgrade all stochastic panels.
gp=SRC/'analytic_geometry_verified.csv'
assert gp.is_file(),gp
g=pd.read_csv(gp);plotdata('Figure1_geometry',g)
fig,aa=plt.subplots(2,2,figsize=(7.01,5.95));fig.subplots_adjust(left=.095,right=.985,bottom=.18,top=.94,wspace=.29,hspace=.52)
a,b,c,d=aa.flat
for m,col,ls in zip(REG,['#444444','#176B87','#9B713C'],['-','--',':']):a.plot(g.utilization,g[m]/g.oracle_best_value_cny,color=col,ls=ls,label=RL[m])
a.set(xlabel='Utilization $u$',ylabel='Oracle mode cost / optimum',ylim=(.985,1.55));a.legend(frameon=False,fontsize=7);panel(a,'a','Tariff geometry, $\u03b1=1.20$')
b.plot(g.utilization,100*g.oracle_margin_cny/g.oracle_best_value_cny,color='#333333')
sub=o[o.alpha==1.2];b.scatter(sub.utilization,100*sub.oracle_gap_cny/sub.oracle_best_value_cny,s=17,c='#333333',zorder=3)
b.set(xlabel='Utilization $u$',ylabel='Oracle mode gap (%)',yscale='log');panel(b,'b','Separation from the next mode')
for m in M:
 z=p[p.method_id==m].set_index('cell_id').loc[CELLS];x=np.arange(9)+(M.index(m)-2)*.12
 c.errorbar(x,z.regret_cny_per_month_mean/1000,yerr=z.regret_cny_per_month_t95_halfwidth/1000,fmt=MK[m],color=C[m],ms=3,capsize=1.4,lw=.7)
 d.errorbar(x,100*z.accuracy,yerr=[100*(z.accuracy-z.accuracy_wilson95_low),100*(z.accuracy_wilson95_high-z.accuracy)],fmt=MK[m],color=C[m],ms=3,capsize=1.4,lw=.7)
for ax in [c,d]:
 group_cells(ax,o)
c.set_ylabel('Mean regret (10³ CNY/month)');c.set_ylim(bottom=0);d.set(ylabel='Oracle-mode accuracy (%)',ylim=(-2,103));panel(c,'c','Mean regret');panel(d,'d','Mode accuracy')
legend_methods(fig,y=.052);plotdata('Figure1_cells',p.merge(o[['cell_id','alpha','utilization']],on='cell_id'));save(fig,'Figure1','main')
# Figure 3: certificate dominates; within-contract loss is conditional on correct selection.
checks=js('primary/primary_analysis_checks.json');correct=js('teacher_sonnet/teacher_fig3d_contract_correct_checks.json')
f3=fig=plt.figure(figsize=(7.01,5.55));gs=fig.add_gridspec(2,2,left=.11,right=.97,bottom=.13,top=.94,hspace=.72,wspace=.34)
a,b,c,d=[fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
count=int(checks['decision_rows']);dm=round(count*checks['d_mv_certificate_coverage']);pw=round(count*checks['pairwise_certificate_coverage'])
for i,m in enumerate(M):
 z=p[p.method_id==m];v=100*z.d_mv_certificate_coverage.mean();w=100*z.pairwise_certificate_coverage.mean();a.plot([v,w],[4-i]*2,color=C[m],lw=1.2);a.plot(v,4-i,marker=MK[m],mfc='white',mec=C[m],ms=5);a.plot(w,4-i,marker=MK[m],color=C[m],ms=5)
a.set_yticks(range(5),M[::-1]);a.set(xlabel='Certified decision fraction (%)',ylim=(-.5,4.7),xlim=(0,32));panel(a,'a','Coverage by method');a.legend(handles=[Line2D([],[],marker='o',mfc='white',mec='#444',lw=0,label='$d_{MV}<1$'),Line2D([],[],marker='o',color='#444',lw=0,label='Pairwise')],loc='lower right',frameon=False,fontsize=7)
for status,marker,col,label in [('d_mv_certificate_coverage','o','#444444','$d_{MV}<1$'),('pairwise_certificate_coverage','s','#176B87','Pairwise gap')]:
 z=p.groupby('cell_id')[status].mean().reindex(CELLS)*100;off=-.12 if status=='d_mv_certificate_coverage' else .12;b.plot(np.arange(9)+off,z,linestyle='none',marker=marker,ms=3.8,color=col,label=label)
group_cells(b,o);b.set(ylabel='Certified fraction (%)',ylim=(0,100));b.legend(frameon=False,loc='upper left');panel(b,'b','Coverage by tariff cell')
for ax,typ,ltr in [(c,'dmv','c'),(d,'pairwise','d')]:
 q=read(f'teacher_sonnet/teacher_fig3d_{typ}_certificate_pooled.csv');plotdata('Figure3_conditional_'+typ,q)
 for i,m in enumerate(M):
  for st,off,fill in [('certified',.13,True),('uncertified',-.13,False)]:
   r=q[(q.method_id==m)&(q.certificate_status==st)].iloc[0];v=r.regret_cny_per_month_mean/1000;l=r.t95_low/1000;h=r.t95_high/1000
   ax.errorbar(v,4-i+off,xerr=[[v-l],[h-v]],fmt=MK[m],mfc=C[m] if fill else 'white',mec=C[m],ecolor=C[m],ms=4,capsize=2,lw=.9)
 ax.set_yticks(range(5),M[::-1] if ax==c else []);ax.set(xlabel='Within-contract regret (10³ CNY/month)',ylim=(-.5,4.6),xlim=(0,20));panel(ax,ltr,'Correct contract: '+('$d_{MV}$ split' if typ=='dmv' else 'pairwise split'))
c.legend(handles=[Line2D([],[],marker='o',color='#444',lw=0,label='Certified'),Line2D([],[],marker='o',mfc='white',mec='#444',lw=0,label='Uncertified')],loc='upper right',frameon=False,fontsize=7)
plotdata('Figure3_coverage',p[['method_id','cell_id','histories','pairwise_certificate_coverage','d_mv_certificate_coverage']]);save(fig,'Figure3','main')

figure_dir=WORKSPACE/'CSSC_SUBMISSION_20261112'/'Manuscript'/'figures'
layout={'status':'PASS' if all(x['passed'] for x in AUD) else 'FAIL','checks':AUD,
        'figures':['Figure1','Figure3'],'source_analysis':str(SRC/'primary/primary_analysis_checks.json'),
        'source_analysis_sha256':hashlib.sha256((SRC/'primary/primary_analysis_checks.json').read_bytes()).hexdigest(),
        'pdf_sha256':{name:hashlib.sha256((figure_dir/name).read_bytes()).hexdigest()
                      for name in ('Figure_1.pdf','Figure_3.pdf')}}
(ROOT/'audit'/'cssc_primary_figures_layout.json').write_text(json.dumps(layout,indent=2),encoding='utf-8')
if layout['status']!='PASS':
 raise AssertionError(AUD)
print('CSSC primary figures updated:',layout['figures'])
