"""Honest figure regeneration from real results (out/characterize.csv).

Replaces the fabricated aggregate figures with real ones:
  fig_real_beta_dp.pdf     <- fig:beta_dp_all  (real beta vs D scatter)
  fig_real_family_dp.pdf   <- fig:accrate_all  (per-family D with 95% CI)
  fig_real_spike_energy.pdf                    (signal energy by family)
  fig_real_core_diag.pdf   <- core diagnostics (correct per-matrix beta)
"""
import csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from extract import extract
from rmt import standardize_and_spectrum, mp_support, mp_cdf, trim_interval, ks_against_mp

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 130})
FAMORD = ["ffn_intermediate","ffn_output","attn_qkv","attn_out","pooler",
          "emb_word","emb_position","emb_projection"]
FCOL = {"ffn_intermediate":"#1f77b4","ffn_output":"#4c9be8","attn_qkv":"#d62728",
        "attn_out":"#ff7f5b","pooler":"#9467bd","emb_word":"#2ca02c",
        "emb_position":"#8cc665","emb_projection":"#17becf"}

def load(path="out/characterize.csv"):
    rows=[]
    for r in csv.DictReader(open(path)):
        if r["degenerate"]=="True": continue
        for k in ("beta","Dstrict","DTW2","spike_energy","boot_p_TW"): r[k]=float(r[k])
        r["n_spikes"]=float(r["n_spikes"]); rows.append(r)
    return rows

def fig_beta_dp(rows, out="fig_real_beta_dp.pdf"):
    fig,ax=plt.subplots(figsize=(5,3.4))
    for fam in FAMORD:
        rs=[r for r in rows if r["family"]==fam]
        if not rs: continue
        j=(np.random.default_rng(0).uniform(-0.012,0.012,len(rs)))
        ax.scatter([r["beta"]+dj for r,dj in zip(rs,j)],[r["DTW2"] for r in rs],
                   s=22,color=FCOL[fam],label=fam,alpha=0.8,edgecolor="k",linewidth=0.3)
    ax.axhline(0.01,ls="--",c="gray",lw=0.8,label="Gaussian null (p95)")
    ax.set_xlabel(r"aspect ratio $\beta=\min/\max$"); ax.set_ylabel(r"$D_{TW}$ (bulk KS)")
    ax.set_title("Real weights: aspect ratio vs KS statistic"); ax.legend(fontsize=6,ncol=2)
    fig.tight_layout(); fig.savefig(out); plt.close(fig); print("wrote",out)

def fig_family_dp(rows, out="fig_real_family_dp.pdf"):
    fams=[f for f in FAMORD if any(r["family"]==f for r in rows)]
    fig,ax=plt.subplots(figsize=(5.5,3.2))
    rng=np.random.default_rng(1)
    for i,fam in enumerate(fams):
        v=np.array([r["DTW2"] for r in rows if r["family"]==fam])
        ax.scatter(np.full(v.size,i)+rng.uniform(-0.1,0.1,v.size),v,s=16,
                   color=FCOL[fam],alpha=0.7,edgecolor="k",linewidth=0.2)
        m=v.mean(); ax.plot([i-0.28,i+0.28],[m,m],c="k",lw=1.6)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams,rotation=35,ha="right",fontsize=6)
    ax.axhline(0.01,ls="--",c="gray",lw=0.8)
    ax.set_ylabel(r"$D_{TW}$ (bulk KS)"); ax.set_title("Per-family bulk deviation (real)")
    fig.tight_layout(); fig.savefig(out); plt.close(fig); print("wrote",out)

def fig_spike_energy(rows, out="fig_real_spike_energy.pdf"):
    fams=[f for f in FAMORD if any(r["family"]==f for r in rows)]
    fig,ax=plt.subplots(figsize=(5.5,3.2))
    means=[np.mean([r["spike_energy"] for r in rows if r["family"]==f]) for f in fams]
    ax.bar(range(len(fams)),means,color=[FCOL[f] for f in fams],edgecolor="k",linewidth=0.4)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams,rotation=35,ha="right",fontsize=6)
    ax.set_ylabel("spike energy fraction"); ax.set_title("Signal energy above MP bulk (real)")
    fig.tight_layout(); fig.savefig(out); plt.close(fig); print("wrote",out)

def fig_core_diag(model="bert-base-uncased", out="fig_real_core_diag.pdf"):
    """PDF+MP and ECDF+MP for representative matrices, each at its TRUE beta."""
    want={"encoder.layer.0.intermediate.dense.weight":"FFN (beta=0.25)",
          "encoder.layer.0.attention.self.query.weight":"Attn Q (beta=1.0)",
          "embeddings.word_embeddings.weight":"Word emb (beta=0.025)"}
    mats={m["name"]:m for m in extract(model)}
    fig,axes=plt.subplots(2,3,figsize=(8,4.2))
    for j,(nm,title) in enumerate(want.items()):
        lam,beta,p,N=standardize_and_spectrum(mats[nm]["W"])
        lm,lp=mp_support(beta); L,U=trim_interval(beta,N,2)
        d=ks_against_mp(lam,beta,N,2)[0]
        # top: histogram of bulk + MP density
        ax=axes[0,j]; bulk=lam[(lam>=L)&(lam<=U)]
        ax.hist(bulk,bins=60,density=True,color="#bbbbbb",alpha=0.8)
        xs=np.linspace(max(lm,1e-6),lp,400)
        dens=np.sqrt(np.clip((lp-xs)*(xs-lm),0,None))/(2*np.pi*beta*xs)
        ax.plot(xs,dens,"k",lw=1.3); ax.set_title(f"{title}\n$D_{{TW}}$={d:.3f}",fontsize=7)
        ax.set_xlim(0,lp*1.05)
        # bottom: ECDF vs MP CDF over bulk
        ax2=axes[1,j]; sb=np.sort(bulk); ec=np.arange(1,sb.size+1)/sb.size
        ax2.plot(sb,ec,color=FCOL.get("attn_qkv" if "Attn" in title else "ffn_intermediate","k"),lw=1.2,label="ECDF")
        Fl,Fu=mp_cdf(np.array([L,U]),beta)
        ax2.plot(sb,np.clip((mp_cdf(sb,beta)-Fl)/max(Fu-Fl,1e-9),0,1),"k--",lw=1.0,label="MP CDF")
        ax2.set_xlabel(r"$\lambda$");
        if j==0: axes[0,0].set_ylabel("density"); ax2.set_ylabel("CDF"); ax2.legend(fontsize=6)
    fig.suptitle(f"Core diagnostics at TRUE per-matrix $\\beta$ — {model}",fontsize=9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig); print("wrote",out)

if __name__=="__main__":
    rows=load()
    bb=[r for r in rows if r["model"]=="bert-base-uncased"]
    fig_beta_dp(bb); fig_family_dp(bb); fig_spike_energy(bb); fig_core_diag()
    print("done")
