import os, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BASE="/mnt/c/Users/MayurGaidhane/Downloads/indic_pipeline_full_dataset/indic_language_pipeline"
RES=os.path.join(BASE,"outputs","results"); FIG=os.path.join(BASE,"outputs","figures")
os.makedirs(FIG,exist_ok=True)
def pick(a,b):
    pa=os.path.join(RES,a); return pa if os.path.exists(pa) else os.path.join(RES,b)
d=pd.read_csv(pick("lmm_table_v7_updated.csv","lmm_table_v7.csv"))
B,O,A="#2a78d6","#eb6834","#1baf7a"
INK,SUB,GRID="#0b0b0b","#52514e","#e3e2de"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.edgecolor":GRID,"axes.labelcolor":SUB,"xtick.color":SUB,"ytick.color":INK})
def get(m,p):
    r=d[(d.Model==m)&(d.Predictor==p)]
    return None if r.empty else r.iloc[0]
def forest(fn,title,preds,labels,series,xlabel="Standardised coefficient (beta), 95% CI",h=None):
    n=len(series); fig,ax=plt.subplots(figsize=(8.2,h or (1.0+1.05*len(preds))),dpi=200)
    off=[(i-(n-1)/2)*0.22 for i in range(n)]
    for j,(m,lab,c) in enumerate(series):
        for i,p in enumerate(preds):
            r=get(m,p)
            if r is None: continue
            y=len(preds)-1-i+off[j]
            ax.plot([r.CI_lo,r.CI_hi],[y,y],color=c,lw=2,solid_capstyle="round",zorder=2)
            ax.plot(r.beta,y,"o",ms=7,color=c,mec="white",mew=1.2,zorder=3,label=lab if i==0 else None)
    ax.axvline(0,color=SUB,lw=1,ls=(0,(4,3)),zorder=1)
    ax.set_yticks(range(len(preds))); ax.set_yticklabels(labels[::-1])
    ax.grid(axis="x",color=GRID,lw=.8); ax.set_axisbelow(True)
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y",length=0)
    ax.set_xlabel(xlabel); ax.set_title(title,loc="left",fontsize=11,color=INK,fontweight="bold")
    ax.legend(frameon=False,loc="lower right",fontsize=9)
    fig.tight_layout(); fig.savefig(fn,facecolor="white"); plt.close(fig)
ling=["IC_z","SC_z","fwd_surp_z","bwd_surp_z","wlen_z","log_freq_z"]
ll=["IC","SC","Forward surprisal","Backward surprisal","Word length","log frequency"]
forest(os.path.join(FIG,"lmm_v7_coefficients_linguistic.png"),"V7 linguistic coefficients: M4 (n = 406,164) vs M6 (n = 329,450)",ling,ll,
  [("M4: Full Linguistic","M4 (all words, n = 406,164)",B),("M6: Full","M6 (acoustic subset, n = 329,450)",O)])
forest(os.path.join(FIG,"lmm_v7_coefficients_by_register.png"),"V7 M4 formula by register",ling[:4],ll[:4],
  [("M4: Conversation","Conversation (n = 122,023)",B),("M4: Extempore","Extempore (n = 262,974)",O),("M4: Read","Read (n = 21,167)",A)],h=4.4)
forest(os.path.join(FIG,"lmm_v7_coefficients_acoustic.png"),"V7 acoustic coefficients: M5 vs M6 (n = 329,450)",["f0_z","rms_z","sr_z","pause_z"],["F0 mean","RMS energy","Speech rate","Pause before"],
  [("M5: Acoustic only","M5 acoustic only",B),("M6: Full","M6 full",O)],h=4.0)
# R2 from lmm_results_v7.txt
models=["M0","M1","M2","M3","M4","M7","M8","M9","M5*","M6*"]
r2m=[0.0000,0.0049,0.0049,0.0049,0.0049,0.0019,0.0049,0.0049,0.0414,0.0415]
r2c=[0.5262,0.5266,0.5264,0.5267,0.5265,0.5259,0.5265,0.5268,0.2835,0.2839]
import numpy as np
fp=os.path.join(RES,"lmm_fit_v7_updated.csv")
if os.path.exists(fp):
    f=pd.read_csv(fp).set_index("Model")
    sel=[("M0","M0: Null"),("M1","M1: Controls"),("M2","M2: DLT only"),("M3","M3: Surprisal"),("M4","M4: Full Ling."),("M7","M7: No log_freq"),("M8","M8: SC alone"),("M9","M9: IC alone"),("M5*","M5: Acoustic"),("M6*","M6: Full")]
    models=[a for a,b in sel if b in f.index]
    r2m=[float(f.loc[b,"R2m"]) for a,b in sel if b in f.index]
    r2c=[float(f.loc[b,"R2c"]) for a,b in sel if b in f.index]
fig,ax=plt.subplots(figsize=(8.2,4.2),dpi=200); x=np.arange(len(models)); w=.38
b1=ax.bar(x-w/2,r2m,w,color=B,label="Marginal R2 (fixed effects)"); b2=ax.bar(x+w/2,r2c,w,color=O,label="Conditional R2 (fixed + random)")
for r,v in zip(b1,r2m): ax.text(r.get_x()+r.get_width()/2,v+.012,f"{v:.3f}",ha="center",fontsize=7.5,color=SUB)
for r,v in zip(b2,r2c): ax.text(r.get_x()+r.get_width()/2,v+.012,f"{v:.2f}",ha="center",fontsize=7.5,color=SUB)
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0,.65); ax.set_ylabel("R2")
ax.grid(axis="y",color=GRID,lw=.8); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.set_title("V7 marginal and conditional R2 (Nakagawa and Schielzeth)",loc="left",fontsize=11,color=INK,fontweight="bold")
ax.text(0,-.19,"* M5 and M6 use the 329,450-word acoustic subset; all others use 406,164 words. Values from lmm_results_v7.txt.",transform=ax.transAxes,fontsize=8,color=SUB)
ax.legend(frameon=False,loc="upper right",fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(FIG,"lmm_v7_r2.png"),facecolor="white"); plt.close(fig)
print("done")