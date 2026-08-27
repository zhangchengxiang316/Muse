#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from gloss.final_model import build_model
from gloss.fullcascade_features import fullcascade_to_model_input
from gloss.temporal_aug import temporal_speed_warp

def readj(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def loadx(p):
    r=np.load(p).astype(np.float32)
    if r.shape==(4,54,64):return r
    if r.shape==(1,4,54,64):return r[0]
    return np.asarray(fullcascade_to_model_input(r,64),dtype=np.float32)
class DS(Dataset):
    def __init__(self,R,s,f):self.R=R;self.s=str(s);self.f=f
    def __len__(self):return len(self.R)
    def __getitem__(self,i):
        x=loadx(self.R[i]["views"][self.s])
        if abs(self.f-1)>1e-9:x=temporal_speed_warp(x,self.f)
        return torch.from_numpy(x),int(self.R[i]["label"])
def logits(o):return o[:,:,0,0] if o.ndim==4 else o
@torch.no_grad()
def ev(m,dl,d):
    cm=np.zeros((6,6),int)
    for x,y in dl:
        p=logits(m(x.to(d))).argmax(1).cpu().numpy()
        for a,b in zip(y.numpy(),p):cm[a,b]+=1
    acc=np.trace(cm)/cm.sum();fs=[]
    for i in range(6):
        tp=cm[i,i];fp=cm[:,i].sum()-tp;fn=cm[i,:].sum()-tp
        pr=tp/(tp+fp) if tp+fp else 0.;rc=tp/(tp+fn) if tp+fn else 0.
        fs.append(2*pr*rc/(pr+rc) if pr+rc else 0.)
    return float(acc),float(np.mean(fs))
def plot(M,ys,xs,title,p):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5));im=ax.imshow(M,aspect="auto")
    ax.set_xticks(range(len(xs)));ax.set_xticklabels(xs);ax.set_yticks(range(len(ys)));ax.set_yticklabels(ys)
    ax.set_xlabel("speed factor");ax.set_ylabel("source stride");ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):ax.text(j,i,f"{M[i,j]*100:.1f}",ha="center",va="center")
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(p,dpi=220,bbox_inches="tight");plt.close(fig)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--groups",type=Path,required=True);ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True);ap.add_argument("--strides",type=int,nargs="+",default=[3,4,5,6])
    ap.add_argument("--speed-factors",type=float,nargs="+",default=[.75,.875,1.,1.125,1.25]);ap.add_argument("--batch-size",type=int,default=64);ap.add_argument("--workers",type=int,default=4)
    a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);R=readj(a.groups);d=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(a.checkpoint,map_location=d);m=build_model(int(ck.get("num_classes",6))).to(d);m.load_state_dict(ck["model"]);m.eval()
    A=np.zeros((len(a.strides),len(a.speed_factors)));F=np.zeros_like(A);rows=[]
    for i,s in enumerate(a.strides):
        for j,f in enumerate(a.speed_factors):
            ac,mf=ev(m,DataLoader(DS(R,s,f),batch_size=a.batch_size,shuffle=False,num_workers=a.workers),d);A[i,j]=ac;F[i,j]=mf
            rows.append({"stride":s,"speed_factor":f,"accuracy":ac,"macro_f1":mf});print(s,f,ac,mf)
    with (a.out_dir/"grid.csv").open("w",newline="",encoding="utf-8") as q:
        w=csv.DictWriter(q,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    S={"mean_accuracy":float(A.mean()),"worst_cell_accuracy":float(A.min()),"mean_macro_f1":float(F.mean()),"worst_cell_macro_f1":float(F.min()),
       "per_stride_mean_accuracy":{str(s):float(A[i].mean()) for i,s in enumerate(a.strides)},
       "per_speed_mean_accuracy":{str(f):float(A[:,j].mean()) for j,f in enumerate(a.speed_factors)}}
    (a.out_dir/"summary.json").write_text(json.dumps(S,indent=2)+"\n");plot(A,a.strides,a.speed_factors,"Accuracy: stride × speed",a.out_dir/"accuracy_heatmap.png");plot(F,a.strides,a.speed_factors,"Macro-F1: stride × speed",a.out_dir/"macro_f1_heatmap.png")
    print(json.dumps(S,indent=2))
if __name__=="__main__":main()
