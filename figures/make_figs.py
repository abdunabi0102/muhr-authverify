import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import poisson
import os; R=os.path.join(os.path.dirname(__file__),"..","results","flat")+"/"; O=os.path.join(os.path.dirname(__file__),"out")+"/"; os.makedirs(O,exist_ok=True)
INK="#1f1f1d"; INK2="#5f5e5a"; GRID="#e4e2dc"
S=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#6250d6","#e34948"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":INK2,"axes.labelcolor":INK,
    "xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
def save(f,n):
    f.savefig(O+n+".pdf",bbox_inches="tight"); f.savefig(O+n+".png",dpi=220,bbox_inches="tight"); plt.close(f)

# ---------- Fig 1: codec and verifier
f,ax=plt.subplots(figsize=(8.2,3.3)); ax.set_xlim(0,100); ax.set_ylim(0,40); ax.axis("off")
def box(x,y,w,h,t,fc="#f3f2ee"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.0",fc=fc,ec=INK2,lw=0.8))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=6.6,color=INK,linespacing=1.25)
def arr(x1,y1,x2,y2): ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",color=INK2,lw=0.8,mutation_scale=7))
W=11.6; G=2.6
ax.text(0,37.5,"Embedding (registrar)",fontsize=8,weight="bold",color=INK)
lab=["Payload P\n48 bits","Tag τ =\nHMAC$_{K2}$(P)\n23 bits","Encrypt P:\nAES-CTR$_{K1}$\nIV = τ","BCH(127,71)\nencode\nC ‖ τ","XOR mask M$_{K3}$,\npermute π$_{K3}$","128-bit\nword w","Host network\nembeds w"]
for i,t in enumerate(lab):
    x=i*(W+G+0.3); box(x,23,W,11,t,fc="#e6effb" if i in (1,2,4) else "#f3f2ee")
    if i: arr(x-G-0.1,28.5,x-0.2,28.5)
ax.text(0,17.5,"Verification (trusted verifier holding K)",fontsize=8,weight="bold",color=INK)
lab2=["Image x′\n(possibly\nattacked)","Decode each\nhypothesis\nh = 1..N$_{geo}$","Un-permute,\nun-mask,\nBCH decode r = 9","Decrypt C\nwith IV = τ,\nrecompute tag","Tags equal?\naccept first h","Return P\nor reject"]
W2=13.8; G2=2.9
for i,t in enumerate(lab2):
    x=i*(W2+G2); box(x,3,W2,11,t,fc="#e6effb" if i in (2,3,4) else "#f3f2ee")
    if i: arr(x-G2+0.2,8.5,x-0.2,8.5)
ax.text(100,0.3,"shaded: steps that use the key",ha="right",fontsize=6.5,color=INK2)
save(f,"fig1_codec")

# ---------- Fig 2: truncated-tag tightness
rows=list(csv.DictReader(open(R+"e4_fa_results.csv")))
from collections import defaultdict
g=defaultdict(lambda:[0,0.0])
for x in rows:
    if x["subset"]!="natural": continue
    for b in (8,10,12,16):
        k=(x["design"],x["whiten"],b); g[k][0]+=int(x[f"acc_{b}"]); g[k][1]+=float(x[f"pred_{b}"])
f,ax=plt.subplots(figsize=(4.2,3.6))
xx=np.logspace(-1,3.2,200)
lo=np.array([poisson.ppf(0.025,m) for m in xx]); hi=np.array([poisson.ppf(0.975,m) for m in xx])
ax.fill_between(xx,np.maximum(lo,0.08),np.maximum(hi,0.08),color="#cde2fb",lw=0,label="95 % Poisson band")
ax.plot(xx,xx,color=S[0],lw=1.5,label="observed = predicted")
des=["A","C","E","F","G8","G9"]; mk=["o","s","D","^","v","P"]
for d,m in zip(des,mk):
    for w,fc in (("1",INK),("0","none")):
        pts=[(g[(d,w,b)][1],g[(d,w,b)][0]) for b in (8,10,12,16) if (d,w,b) in g and g[(d,w,b)][1]>=0.3]
        if not pts: continue
        p=np.array(pts,float); y=np.where(p[:,1]==0,0.12,p[:,1])
        ax.scatter(p[:,0],y,marker=m,s=26,facecolors=fc,edgecolors=INK,lw=0.8,zorder=3,label=f"{d}" if w=="1" else None)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.25,1600); ax.set_ylim(0.09,1600)
ax.set_xlabel("Predicted acceptances  (words × p$_{BCH}$ × 2$^{-b}$)"); ax.set_ylabel("Observed acceptances")
ax.grid(color=GRID,lw=0.6); ax.set_axisbelow(True)
ax.axhline(0.12,color=GRID,lw=0.6); ax.text(0.27,0.135,"0 observed",fontsize=6.5,color=INK2)
ax.legend(fontsize=6.6,frameon=False,loc="upper left",ncol=2,title="filled: whitened; open: not",title_fontsize=6.6)
save(f,"fig2_tightness")

# ---------- Fig 3: calibration ratio at alpha = 1e-3
cal=[x for x in csv.DictReader(open(R+"e4c_calibration.csv")) if x["subset"]=="natural"]
meth=["wam","trustmark","pixelseal","hidden","videoseal","cin","mbrs"]
names={"wam":"WAM","trustmark":"TrustMark-Q","pixelseal":"Pixel Seal","hidden":"HiDDeN","videoseal":"Video Seal","cin":"CIN","mbrs":"MBRS"}
def get(m,ref,col): return float(next(x for x in cal if x["method"]==m and x["reference"]==ref)[col])
series=[("Random message, single decode","random","fpr_native_0.001",S[0],"o"),("Random message, search","random","fpr_search_0.001",S[0],"o"),
        ("Adversarial message, single decode","majority","fpr_native_0.001",S[1],"s"),("Adversarial message, search","majority","fpr_search_0.001",S[1],"s")]
f,ax=plt.subplots(figsize=(6.4,3.2)); y=np.arange(len(meth))
for i,(lab,ref,col,c,mk) in enumerate(series):
    v=np.array([max(get(m,ref,col),1e-5)/1e-3 for m in meth]); off=(i-1.5)*0.17
    filled="search" in lab
    ax.scatter(v,y+off,marker=mk,s=30,facecolors=c if filled else "white",edgecolors=c,lw=1.2,zorder=3,label=lab)
ax.axvline(1,color=INK2,lw=1,ls="--"); ax.text(1.08,len(meth)-0.45,"nominal",fontsize=7,color=INK2)
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([names[m] for m in meth]); ax.invert_yaxis()
ax.set_xlabel("Observed FPR / nominal FPR  (nominal 10$^{-3}$, 5 000 unmarked COCO images)")
ax.grid(axis="x",color=GRID,lw=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=7,frameon=False,loc="upper center",bbox_to_anchor=(0.45,-0.2),ncol=2)
save(f,"fig3_calibration")

# ---------- Fig 4: verification heatmap
ver=list(csv.DictReader(open(R+"e3_coco_verification.csv")))
cod=list(csv.DictReader(open(R+"codec_on_baselines.csv")))
EN={"yo'q":"none","JPEG":"JPEG","WebP":"WebP","Gauss shovqini":"Gaussian noise","Xiralashtirish":"Gaussian blur","Median filtr":"Median",
    "Kesish":"Crop to","Masshtablash":"Scale","Burish":"Rotate","Gamma":"Gamma","Gistogramma tekislash":"Histogram eq.","Ekran surati":"Screenshot","JPEG50+kesish0.7":"JPEG50 + crop 70%"}
conds=[]; colA={};colG={}
for x in ver:
    if x["policy"]!="search": continue
    k=(x["attack"],x["param"])
    if k not in conds: conds.append(k)
    if x["variant"]=="A": colA[k]=float(x["rate"])
    if x["variant"]=="G9": colG[k]=float(x["rate"])
cP={(x["attack"],x["param"]):float(x["verified"]) for x in cod if x["method"]=="pixelseal" and x["decode"]=="G9"}
cV={(x["attack"],x["param"]):float(x["verified"]) for x in cod if x["method"]=="videoseal" and x["decode"]=="G9"}
def lab(k):
    a,p=k; n=EN.get(a,a)
    if a=="Kesish": return f"Crop to {int(float(p)*100)} %"
    if a=="Burish": return f"Rotate {p}°"
    if a=="Masshtablash": return f"Scale {p}"
    return f"{n} {p}".strip()
M=np.array([[colA[k],colG[k],cP.get(k,np.nan),cV.get(k,np.nan)] for k in conds])*100
cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b",["#f3f6fb","#9ec5f4","#2a78d6","#104281"])
f,ax=plt.subplots(figsize=(4.6,7.4))
ax.imshow(M,cmap=cmap,vmin=0,vmax=100,aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v=M[i,j]; ax.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=6.8,color="white" if v>60 else INK)
ax.set_xticks(range(4)); ax.set_xticklabels(["Ours\nA","Ours\nG9","Pixel Seal\n+ G9","Video Seal\n+ G9"],fontsize=7.5)
ax.xaxis.tick_top(); ax.set_yticks(range(len(conds))); ax.set_yticklabels([lab(k) for k in conds],fontsize=7.2)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-.5,4,1),minor=True); ax.set_yticks(np.arange(-.5,len(conds),1),minor=True)
ax.grid(which="minor",color="white",lw=1.5)
ax.set_title("Authenticated verification (%)",fontsize=8.5,pad=26,color=INK)
save(f,"fig4_verification")
print("ok", len(conds))
