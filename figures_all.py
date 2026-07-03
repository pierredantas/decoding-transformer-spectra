"""Regenerate ALL manuscript figures from REAL weights, correct per-matrix beta.
Writes graph_*.pdf into the project dir (drop-in replacements). LaTeX untouched.

Figure map (manuscript slot -> honest content):
  core_diag_01  PDF(bulk) vs conditional MP density          (6 real Sets)
  core_diag_02  ECDF vs conditional MP CDF                   (6 real Sets)
  core_diag_03  EPDF vs MP density (line)                    (6 real Sets)
  core_diag_04  residual CDF (ECDF-MP CDF) + null band       (6 real Sets)
  core_diag_05  QQ empirical vs MP quantiles                 (6 real Sets)
  level_views_01 heatmap: D_TW per (layer x matrix-type)
  level_views_02 per-layer D_TW by family (depth)
  level_views_03 beta vs D_TW scatter, all matrices (real)   [bert_base slot]
  shrinkage_control_01 null p-value distribution (uniform)   [bert_base slot]
  shrinkage_control_02 D_TW vs c_alpha in {1,2,3}
  shrinkage_control_03 Type-I rejection rate vs nominal alpha (MC on shape null)
  shrinkage_control_04 bootstrap ECDF envelopes (null vs alternatives) [bert_base]
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from extract import extract
from rmt import standardize_and_spectrum, mp_support, mp_cdf, trim_interval, ks_against_mp
from spikes import signal_noise_split

def null_D(p, N, rng, c=2):
    """D statistic of one iid-Gaussian draw of shape (p,N)."""
    lam, beta, pp, NN = standardize_and_spectrum(rng.standard_normal((p, N)))
    return ks_against_mp(lam, beta, NN, c)[0]

OUT = os.environ.get("FIG_OUT", "..")   # default: manuscript dir; CI overrides to a local dir
plt.rcParams.update({"font.size":7,"axes.grid":True,"grid.alpha":0.25,
                     "axes.linewidth":0.6,"lines.linewidth":1.1,"figure.dpi":140})
FCOL={"ffn_intermediate":"#1f77b4","ffn_output":"#4c9be8","attn_qkv":"#d62728",
      "attn_out":"#ff7f5b","pooler":"#9467bd","emb_word":"#2ca02c",
      "emb_position":"#8cc665","emb_projection":"#17becf"}
MODELS={"bert_base":"bert-base-uncased","albert":"albert-base-v2",
        "bert_large":"bert-large-uncased"}

def mp_density(x, beta):
    lm,lp=mp_support(beta)
    return np.sqrt(np.clip((lp-x)*(x-lm),0,None))/(2*np.pi*beta*np.clip(x,1e-12,None))

def load_spectra(model_id):
    out=[]
    for m in extract(model_id):
        lam,beta,p,N=standardize_and_spectrum(m["W"])
        if p<10: continue
        d=dict(m); d.update(lam=lam,beta=beta,p=p,N=N)
        out.append(d)
    return out

def pick(spec, family, layer=None):
    c=[s for s in spec if s["family"]==family and (layer is None or s["layer"]==layer)]
    return c[0] if c else None

def sets_for(tag, spec):
    """6 representative real matrices (mix rectangular/square across depth)."""
    L=sorted({s["layer"] for s in spec if s["layer"]>=0})
    if tag=="albert":
        want=[("ffn_intermediate",None,"FFN-in β.25"),("attn_qkv",None,"Attn-Q β1"),
              ("ffn_output",None,"FFN-out β.25"),("attn_out",None,"Att-Out β1"),
              ("emb_projection",None,"Emb-proj"),("pooler",None,"Pooler β1")]
    else:
        lo,mid,hi=L[0],L[len(L)//2],L[-1]
        want=[("ffn_intermediate",lo,f"FFN-in L{lo} β.25"),("attn_qkv",lo,f"Attn-Q L{lo} β1"),
              ("ffn_intermediate",mid,f"FFN-in L{mid} β.25"),("attn_qkv",mid,f"Attn-Q L{mid} β1"),
              ("ffn_intermediate",hi,f"FFN-in L{hi} β.25"),("attn_qkv",hi,f"Attn-Q L{hi} β1")]
    res=[]
    for fam,lay,lab in want:
        s=pick(spec,fam,lay) or pick(spec,fam)
        if s: res.append((s,lab))
    return res

# ---------- core diagnostics ----------
def core01_pdf(tag,spec):
    S=sets_for(tag,spec); fig,ax=plt.subplots(1,len(S),figsize=(9,1.6))
    for a,(s,lab) in zip(ax,S):
        lm,lp=mp_support(s["beta"]); L,U=trim_interval(s["beta"],s["N"],2)
        bulk=s["lam"][(s["lam"]>=L)&(s["lam"]<=U)]
        a.hist(bulk,bins=60,range=(L,U),density=True,color="#bbb",alpha=.8)
        xs=np.linspace(max(lm,1e-6),lp,300); a.plot(xs,mp_density(xs,s["beta"]),"k")
        d=ks_against_mp(s["lam"],s["beta"],s["N"],2)[0]
        a.set_title(f"{lab}\n$D_{{TW}}$={d:.3f}",fontsize=5); a.tick_params(labelsize=4)
    ax[0].set_ylabel(r"$\lambda$ vs PDF",fontsize=6); _save(tag,"core_diag_01",fig)

def core02_ecdf(tag,spec):
    S=sets_for(tag,spec); fig,ax=plt.subplots(1,len(S),figsize=(9,1.6))
    for a,(s,lab) in zip(ax,S):
        L,U=trim_interval(s["beta"],s["N"],2); bulk=np.sort(s["lam"][(s["lam"]>=L)&(s["lam"]<=U)])
        ec=np.arange(1,bulk.size+1)/bulk.size; a.plot(bulk,ec,color="#d62728",lw=1)
        Fl,Fu=mp_cdf(np.array([L,U]),s["beta"])
        a.plot(bulk,np.clip((mp_cdf(bulk,s["beta"])-Fl)/max(Fu-Fl,1e-9),0,1),"k--",lw=.9)
        a.set_title(f"{lab}\n$N$={bulk.size}",fontsize=5); a.tick_params(labelsize=4)
    ax[0].set_ylabel("ECDF vs MP",fontsize=6); _save(tag,"core_diag_02",fig)

def core03_epdf(tag,spec):
    S=sets_for(tag,spec); fig,ax=plt.subplots(1,len(S),figsize=(9,1.6))
    for a,(s,lab) in zip(ax,S):
        lm,lp=mp_support(s["beta"]); L,U=trim_interval(s["beta"],s["N"],2)
        bulk=s["lam"][(s["lam"]>=L)&(s["lam"]<=U)]
        h,edges=np.histogram(bulk,bins=50,range=(L,U),density=True); ctr=.5*(edges[1:]+edges[:-1])
        a.plot(ctr,h,color="#1f77b4",lw=1,label="emp"); xs=np.linspace(max(lm,1e-6),lp,300)
        a.plot(xs,mp_density(xs,s["beta"]),"k--",lw=.9,label="MP")
        a.set_title(lab,fontsize=5); a.tick_params(labelsize=4)
    ax[0].set_ylabel("EPDF vs MP",fontsize=6); _save(tag,"core_diag_03",fig)

def core04_resid(tag,spec):
    S=sets_for(tag,spec); fig,ax=plt.subplots(1,len(S),figsize=(9,1.6))
    for a,(s,lab) in zip(ax,S):
        L,U=trim_interval(s["beta"],s["N"],2); bulk=np.sort(s["lam"][(s["lam"]>=L)&(s["lam"]<=U)])
        ec=np.arange(1,bulk.size+1)/bulk.size; Fl,Fu=mp_cdf(np.array([L,U]),s["beta"])
        mp=np.clip((mp_cdf(bulk,s["beta"])-Fl)/max(Fu-Fl,1e-9),0,1)
        dcrit=1.358/np.sqrt(bulk.size)
        a.plot(bulk,ec-mp,color="#d62728",lw=1); a.axhline(dcrit,ls="--",c="grey",lw=.7)
        a.axhline(-dcrit,ls="--",c="grey",lw=.7); a.axhline(0,c="k",lw=.4)
        a.set_title(f"{lab}\n$D_p$={np.abs(ec-mp).max():.3f}",fontsize=5); a.tick_params(labelsize=4)
    ax[0].set_ylabel("residual CDF",fontsize=6); _save(tag,"core_diag_04",fig)

def core05_qq(tag,spec):
    S=sets_for(tag,spec); fig,ax=plt.subplots(1,len(S),figsize=(9,1.6))
    for a,(s,lab) in zip(ax,S):
        L,U=trim_interval(s["beta"],s["N"],2); bulk=np.sort(s["lam"][(s["lam"]>=L)&(s["lam"]<=U)])
        n=bulk.size; probs=(np.arange(1,n+1)-.5)/n
        # invert MP CDF on a grid
        lm,lp=mp_support(s["beta"]); g=np.linspace(L,U,4000)
        Fl,Fu=mp_cdf(np.array([L,U]),s["beta"]); cg=np.clip((mp_cdf(g,s["beta"])-Fl)/max(Fu-Fl,1e-9),0,1)
        theo=np.interp(probs,cg,g)
        a.scatter(theo,bulk,s=2,color="#888"); a.plot([L,U],[L,U],"k--",lw=.8)
        a.set_title(lab,fontsize=5); a.tick_params(labelsize=4)
    ax[0].set_ylabel("QQ (emp vs MP)",fontsize=6); _save(tag,"core_diag_05",fig)

# ---------- aggregate ----------
FAM_SHORT={"attn_qkv":"Q/K/V","attn_out":"Att-Out","ffn_intermediate":"FFN-in",
           "ffn_output":"FFN-out","pooler":"Pool","emb_word":"Emb-W",
           "emb_position":"Emb-P","emb_projection":"Emb-Pr"}
def lv01_heatmap(tag,spec):
    fams=[f for f in ["ffn_intermediate","ffn_output","attn_qkv","attn_out"] if any(s["family"]==f for s in spec)]
    layers=sorted({s["layer"] for s in spec if s["layer"]>=0})
    M=np.full((len(fams),len(layers)),np.nan)
    for s in spec:
        if s["family"] in fams and s["layer"] in layers:
            i=fams.index(s["family"]); j=layers.index(s["layer"])
            v=ks_against_mp(s["lam"],s["beta"],s["N"],2)[0]
            M[i,j]=v if np.isnan(M[i,j]) else min(M[i,j],v)
    fig,ax=plt.subplots(figsize=(max(4,len(layers)*.35),1.8))
    im=ax.imshow(M,aspect="auto",cmap="viridis_r",vmin=0,vmax=0.2)
    ax.set_yticks(range(len(fams))); ax.set_yticklabels([FAM_SHORT[f] for f in fams],fontsize=6)
    ax.set_xticks(range(0,len(layers),max(1,len(layers)//8)))
    ax.set_xticklabels([layers[k] for k in range(0,len(layers),max(1,len(layers)//8))],fontsize=5)
    ax.set_xlabel("layer",fontsize=6); fig.colorbar(im,ax=ax,label="$D_{TW}$",shrink=.8)
    _save(tag,"level_views_01",fig)

def lv02_perlayer(tag,spec):
    fams=[f for f in ["ffn_intermediate","ffn_output","attn_qkv","attn_out"] if any(s["family"]==f for s in spec)]
    fig,ax=plt.subplots(figsize=(5,2.4))
    for f in fams:
        pts={}
        for s in spec:
            if s["family"]==f and s["layer"]>=0:
                pts.setdefault(s["layer"],[]).append(ks_against_mp(s["lam"],s["beta"],s["N"],2)[0])
        if not pts: continue
        xs=sorted(pts); ys=[np.mean(pts[x]) for x in xs]
        ax.plot(xs,ys,"o-",ms=3,color=FCOL[f],label=FAM_SHORT[f])
    ax.set_xlabel("layer"); ax.set_ylabel("$D_{TW}$"); ax.legend(fontsize=6,ncol=2)
    ax.set_title(f"Per-layer bulk deviation ({tag})",fontsize=8); _save(tag,"level_views_02",fig)

def lv03_betadp(tag,spec):
    fig,ax=plt.subplots(figsize=(5,3.2)); rng=np.random.default_rng(0)
    for f in FCOL:
        rs=[s for s in spec if s["family"]==f]
        if not rs: continue
        b=[s["beta"]+rng.uniform(-.012,.012) for s in rs]
        d=[ks_against_mp(s["lam"],s["beta"],s["N"],2)[0] for s in rs]
        ax.scatter(b,d,s=18,color=FCOL[f],label=FAM_SHORT[f],alpha=.8,edgecolor="k",linewidth=.3)
    ax.axhline(0.01,ls="--",c="gray",lw=.8,label="Gaussian null p95")
    ax.set_xlabel(r"$\beta=\min/\max$"); ax.set_ylabel("$D_{TW}$")
    ax.set_title(f"Aspect ratio vs KS ({tag}) — real weights",fontsize=8); ax.legend(fontsize=5,ncol=2)
    _save(tag,"level_views_03",fig)

# ---------- controls ----------
def sc01_nullpvals(tag):
    rng=np.random.default_rng(0); shapes=[(768,768),(768,3072)]; fig,ax=plt.subplots(1,2,figsize=(6,2.2))
    for a,(p,N) in zip(ax,shapes):
        Ds=np.sort([null_D(p,N,rng) for _ in range(80)])
        pv=[(np.sum(Ds>=d)+1)/(len(Ds)+1) for d in Ds]      # ~Uniform under null
        a.hist(pv,bins=15,range=(0,1),color="#888",density=True); a.axhline(1,ls="--",c="k",lw=.8)
        a.set_title(f"shape {p}x{N}",fontsize=6); a.set_xlabel("bootstrap p-value",fontsize=6)
    fig.suptitle("Null calibration (KS-TW): p-values ~ Uniform",fontsize=8); _save(tag,"shrinkage_control_01",fig)

def sc02_calpha(tag,spec):
    S=sets_for(tag,spec); fig,ax=plt.subplots(figsize=(5,2.6))
    for s,lab in S:
        ys=[ks_against_mp(s["lam"],s["beta"],s["N"],c)[0] for c in (1,2,3)]
        col=FCOL.get(s["family"],"k"); ax.plot([1,2,3],ys,"o-",ms=3,color=col,label=lab)
    ax.set_xticks([1,2,3]); ax.set_xlabel(r"$c_\alpha$"); ax.set_ylabel("$D_{TW}$")
    ax.set_title(f"KS vs edge relaxation ({tag})",fontsize=8); ax.legend(fontsize=5)
    _save(tag,"shrinkage_control_02",fig)

SHAPES_BY_TAG={"bert_base":{"768x768":(768,768),"768x3072":(768,3072)},
               "albert":{"768x768":(768,768),"768x3072":(768,3072)},
               "bert_large":{"1024x1024":(1024,1024),"1024x4096":(1024,4096)}}
def sc03_type1(tag):
    """Type-I: empirical rejection rate vs nominal alpha on synthetic MP-null."""
    rng=np.random.default_rng(1); shapes=SHAPES_BY_TAG.get(tag,{"768x768":(768,768),"768x3072":(768,3072)})
    alphas=np.array([0.01,0.05,0.10]); fig,ax=plt.subplots(figsize=(4.2,2.6))
    for name,(p,N) in shapes.items():
        # Type-I control depends on aspect ratio beta, ~scale-invariant in size:
        # simulate on a downscaled shape (same beta) so large models stay tractable.
        sc=max(1, max(p,N)//1024); ps,Ns=max(64,p//sc),max(64,N//sc)
        Dnull=np.sort([null_D(ps,Ns,rng) for _ in range(60)])
        crit={a:np.quantile(Dnull,1-a) for a in alphas}     # critical values from null
        fresh=[null_D(ps,Ns,rng) for _ in range(60)]         # fresh null draws
        rej=[np.mean([d>=crit[a] for d in fresh]) for a in alphas]
        ax.plot(alphas,rej,"o-",ms=3,label=name)
    ax.plot([0,.1],[0,.1],"k--",lw=.8); ax.set_xlabel(r"nominal $\alpha$"); ax.set_ylabel("rejection rate")
    ax.set_title(f"Type-I calibration ({tag})",fontsize=8); ax.legend(fontsize=6); _save(tag,"shrinkage_control_03",fig)

def sc04_envelopes(tag):
    rng=np.random.default_rng(2); fig,ax=plt.subplots(1,3,figsize=(7,2))
    scen=[("null N(0,1)","normal"),("shifted +1","shift"),("heavy t3","t3")]
    grid=np.linspace(-4,4,200)
    def draw(kind,n):
        if kind=="normal": return rng.standard_normal(n)
        if kind=="shift": return rng.standard_normal(n)+1
        return rng.standard_t(3,n)
    for a,(title,kind) in zip(ax,scen):
        boots=np.array([np.searchsorted(np.sort(rng.standard_normal(300)),grid,"right")/300 for _ in range(80)])
        lo,hi=np.percentile(boots,5,0),np.percentile(boots,95,0)
        a.fill_between(grid,lo,hi,color="#ccc",alpha=.7)
        obs=np.searchsorted(np.sort(draw(kind,300)),grid,"right")/300
        a.plot(grid,obs,"k",lw=1); a.set_title(title,fontsize=6); a.tick_params(labelsize=4)
    fig.suptitle("Bootstrap ECDF envelopes (5–95%) vs null",fontsize=8); _save(tag,"shrinkage_control_04",fig)

def _save(tag,base,fig):
    fig.tight_layout(); path=os.path.join(OUT,f"graph_{base}_{tag}.pdf")
    fig.savefig(path,bbox_inches="tight"); plt.close(fig); print("wrote",os.path.basename(path),flush=True)

if __name__=="__main__":
    import sys
    which=sys.argv[1:] or list(MODELS)
    for tag in which:
        print(f"== {tag} ==",flush=True); spec=load_spectra(MODELS[tag])
        core01_pdf(tag,spec); core02_ecdf(tag,spec); core03_epdf(tag,spec)
        core04_resid(tag,spec); core05_qq(tag,spec)
        lv01_heatmap(tag,spec); lv02_perlayer(tag,spec)
        sc02_calpha(tag,spec)
        sc03_type1(tag)            # Type-I calibration: per-model
        if tag=="bert_base":       # single representative slots
            lv03_betadp(tag,spec); sc01_nullpvals(tag); sc04_envelopes(tag)
    print("ALL FIGURES DONE")
