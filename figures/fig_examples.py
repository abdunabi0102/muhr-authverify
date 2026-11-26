import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import os, sys; D=(sys.argv[1] if len(sys.argv)>1 else "e10_examples")+"/"
O=os.path.join(os.path.dirname(__file__),"out")+"/"; os.makedirs(O,exist_ok=True)
INK="#1f1f1d"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"pdf.fonttype":42})
R={(r["index"],r["host"]):r for r in json.load(open(D+"examples.json"))}
IDS=[23,41,96]; GAIN=10; SIDE=360
def load(i,h):
    return np.asarray(Image.open(f"{D}{i:04d}_{h}.png").convert("RGB")).astype(np.float32)
def crop(a):
    h,w=a.shape[:2]; s=min(h,w); y=(h-s)//2; x=(w-s)//2
    return a[y:y+s,x:x+s]
def show(a):
    return np.asarray(Image.fromarray(np.clip(a,0,255).astype(np.uint8)).resize((SIDE,SIDE),Image.Resampling.LANCZOS))
cols=["Cover","Ours + G9\n(marked)","Ours + G9\n|residual| ×10","Pixel Seal + G9\n|residual| ×10","Video Seal + G9\n|residual| ×10"]
fig,ax=plt.subplots(len(IDS),5,figsize=(7.1,4.55))
for r,i in enumerate(IDS):
    c=crop(load(i,"cover"))
    ims=[show(c),show(crop(load(i,"ours")))]
    for h in ["ours","pixelseal","videoseal"]:
        d=np.abs(crop(load(i,h))-c).mean(2)*GAIN
        g=255-np.clip(d,0,255); ims.append(np.repeat(show(g)[...,None] if show(g).ndim==2 else show(g),1,axis=-1) if False else show(np.stack([g]*3,-1)))
    for k,im in enumerate(ims):
        a=ax[r,k]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_color("#c9c7c0"); s.set_linewidth(0.5)
        if r==0: a.set_title(cols[k],fontsize=6.6,color=INK,pad=3)
        host={1:"ours",2:"ours",3:"pixelseal",4:"videoseal"}.get(k)
        if host and k!=1:
            q=R[(i,host)]
            a.set_xlabel(f"{q['psnr']:.1f} dB · LPIPS {q['lpips']:.3f}",fontsize=6.2,color=INK,labelpad=2)
fig.subplots_adjust(left=0.01,right=0.99,top=0.925,bottom=0.04,wspace=0.04,hspace=0.17)
fig.savefig(O+"fig8_examples.pdf",dpi=300); fig.savefig(O+"fig8_examples.png",dpi=200)
print("ok")
