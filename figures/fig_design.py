import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import comb
import os; S_=os.path.join(os.path.dirname(__file__),"..","results")+"/"
O=os.path.join(os.path.dirname(__file__),"out")+"/"; os.makedirs(O,exist_ok=True)
INK="#1f1f1d"; INK2="#5f5e5a"; GRID="#e4e2dc"
S=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#6250d6","#e34948"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.edgecolor":INK2,"axes.labelcolor":INK,
    "xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
e=pd.read_csv(S_+"e3_coco/errors_best_hyp.csv")
G=e[e.design=="G"]
def ver(r): return G[f"P(e<={r})"].mean()*100
def pb(n,k,r): return 2.0**(k-n)*sum(comb(n,i) for i in range(r+1))
fig,(a,b)=plt.subplots(1,2,figsize=(7.1,2.75),gridspec_kw={"width_ratios":[1.15,1]})
# (a) design space
rs=list(range(4,10))
fa=[pb(127,71,r)*2**-23 for r in rs]; vv=[ver(r) for r in rs]
a.plot(fa,vv,color=GRID,lw=1.4,zorder=1)
a.scatter(fa,vv,s=10,color=INK2,zorder=2)
for r,x,y in zip(rs,fa,vv):
    if r in (4,5,6,7): a.annotate(f"r={r}",(x,y),xytext=(3,-9),textcoords="offset points",fontsize=6,color=INK2)
D={"A":(3.61e-2*2**-32,86.12,True),"C":(6.84e-3*2**-25,ver(5),False),"E":(1.24e-3*2**-29,ver(6),False),
   "F":(1.68e-4*2**-26,ver(7),False),"G8":(1.99e-5*2**-23,89.97,True),"G9":(2.66e-4*2**-23,90.59,True)}
for i,(d,(x,y,meas)) in enumerate(D.items()):
    a.scatter([x],[y],s=34,marker="o",facecolors=S[i] if meas else "white",edgecolors=S[i],lw=1.4,zorder=4)
    off={"A":(7,-3),"C":(7,-3),"E":(-15,-3),"F":(-15,-3),"G8":(-19,-3),"G9":(7,-3)}[d]
    a.annotate(d,(x,y),xytext=off,textcoords="offset points",fontsize=7,color=INK,weight="bold" if d=="G9" else None)
a.set_xscale("log"); a.set_xlabel("False-acceptance bound, $N_{geo}$ = 1"); a.set_ylabel("Mean verification, 27 conditions (%)")
a.set_ylim(84.6,91.4); a.set_xlim(3e-16,1e-9); a.grid(color=GRID,lw=0.5); a.set_axisbelow(True)
from matplotlib.lines import Line2D
a.legend(handles=[Line2D([],[],marker="o",ls="",mfc=INK2,mec=INK2,ms=5,label="measured"),
                  Line2D([],[],marker="o",ls="",mfc="white",mec=INK2,ms=5,label="predicted from errors"),
                  Line2D([],[],color=GRID,lw=1.4,marker=".",mfc=INK2,mec=INK2,label="(127,71), 23-bit tag, radius r")],
         fontsize=6.2,frameon=False,loc="lower left")
a.set_title("(a) Operating points",fontsize=8,loc="left",color=INK)
# (b) error CDF per condition
C=[("JPEG",30,"JPEG 30"),("WebP",50,"WebP 50"),("Gistogramma tekislash",None,"Hist. equalization"),
   ("JPEG50+kesish0.7",None,"JPEG 50 + crop 70%"),("Kesish",0.6,"Crop to 60%"),("Burish",15,"Rotation 15°")]
R=list(range(0,13))
for i,(an,p,lab) in enumerate(C):
    row=G[(G.attack==an)&((G.param.isna()) if p is None else (G.param.astype(float)==float(p)))]
    assert len(row)==1,(an,p)
    b.plot(R,[row[f"P(e<={r})"].iloc[0]*100 for r in R],color=S[i],lw=1.4,label=lab)
b.plot(R,[ver(r) for r in R],color=INK,lw=2,label="Mean, 27 conditions")
for r,t in ((4,"A"),(9,"G9")):
    b.axvline(r,color=INK2,lw=0.8,ls="--"); b.text(r+0.15,4,t,fontsize=7,color=INK2,bbox=dict(fc="white",ec="none",pad=0.5))
b.set_xlabel("Decoding radius r (bit errors corrected)"); b.set_ylabel("Words decodable (%)")
b.set_ylim(0,102); b.set_xlim(0,12); b.grid(color=GRID,lw=0.5); b.set_axisbelow(True)
b.legend(fontsize=6,frameon=False,loc="center right",bbox_to_anchor=(1.0,0.38))
b.set_title("(b) Bit errors per condition",fontsize=8,loc="left",color=INK)
fig.tight_layout()
fig.savefig(O+"fig7_design.pdf",bbox_inches="tight"); fig.savefig(O+"fig7_design.png",dpi=220,bbox_inches="tight")
print({d:round(v[1],2) for d,v in D.items()})
