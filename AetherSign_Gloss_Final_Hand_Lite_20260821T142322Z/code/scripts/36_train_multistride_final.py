#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from gloss.final_model import build_model
from gloss.fullcascade_features import fullcascade_to_model_input
from gloss.fullcascade_dataset import augment_model_input
from gloss.temporal_aug import augment_temporal

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def readj(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def loadx(path):
    raw=np.load(path).astype(np.float32)
    if raw.shape==(4,54,64):return raw
    if raw.shape==(1,4,54,64):return raw[0]
    x=fullcascade_to_model_input(raw,64)
    if x.shape!=(4,54,64):raise ValueError((path,x.shape))
    return x

def parse_probs(xs):
    d={str(int(k)):float(v) for k,v in (x.split(":") for x in xs)}
    z=sum(d.values());return {k:v/z for k,v in d.items()}

class GroupDS(Dataset):
    def __init__(self,p,train,probs=None,eval_stride="4",seed=0,speed_prob=.7,speed_min=.85,speed_max=1.18,jitter=3,base_aug=True):
        self.r=readj(p);self.train=train;self.p=probs;self.es=str(eval_stride)
        self.seed=seed;self.rng=np.random.default_rng(seed)
        self.kw=dict(speed_prob=speed_prob,speed_min=speed_min,speed_max=speed_max,jitter=jitter)
        self.base_aug=base_aug
    def __len__(self):return len(self.r)
    def __getitem__(self,i):
        r=self.r[i];views=r["views"]
        if self.train:
            ks=[k for k in self.p if k in views];ps=np.array([self.p[k] for k in ks]);ps/=ps.sum()
            k=str(self.rng.choice(ks,p=ps))
        else:k=self.es
        x=loadx(views[k])
        if self.train:
            x=augment_temporal(x,self.rng,**self.kw)
            if self.base_aug:x=augment_model_input(x)
        return torch.from_numpy(x),torch.tensor(int(r["label"]),dtype=torch.long)

def worker_seed(worker_id):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed);random.seed(seed)

def logits(o):return o[:,:,0,0] if o.ndim==4 else o

@torch.no_grad()
def evaluate(model,loader,device,num_classes=6):
    model.eval();crit=nn.CrossEntropyLoss();total=correct=0;loss_sum=0.;cm=torch.zeros((num_classes,num_classes),dtype=torch.long)
    for x,y in loader:
        x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);z=logits(model(x));loss=crit(z,y);p=z.argmax(1)
        total+=y.numel();correct+=(p==y).sum().item();loss_sum+=loss.item()*y.numel()
        for t,q in zip(y.cpu(),p.cpu()):cm[int(t),int(q)]+=1
    fs=[]
    A=cm.numpy()
    for i in range(num_classes):
        tp=A[i,i];fp=A[:,i].sum()-tp;fn=A[i,:].sum()-tp
        pr=tp/(tp+fp) if tp+fp else 0.;rc=tp/(tp+fn) if tp+fn else 0.
        fs.append(2*pr*rc/(pr+rc) if pr+rc else 0.)
    return {"loss":loss_sum/max(total,1),"acc":correct/max(total,1),"macro_f1":float(np.mean(fs)),"confusion":cm.tolist()}

def recal_bn(model,loader,device):
    bns=[m for m in model.modules() if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d))]
    if not bns:return
    saved=[m.momentum for m in bns]
    for m in bns:m.reset_running_stats();m.momentum=None
    model.train()
    with torch.no_grad():
        for x,_ in loader:model(x.to(device,non_blocking=True))
    for m,v in zip(bns,saved):m.momentum=v

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-groups",type=Path,required=True);ap.add_argument("--val-groups",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True);ap.add_argument("--num-classes",type=int,default=6)
    ap.add_argument("--epochs",type=int,default=80);ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--lr",type=float,default=1e-3);ap.add_argument("--weight-decay",type=float,default=1e-4)
    ap.add_argument("--label-smoothing",type=float,default=.04);ap.add_argument("--seed",type=int,default=20260819)
    ap.add_argument("--num-workers",type=int,default=4);ap.add_argument("--bn-recalibrate",action="store_true")
    ap.add_argument("--no-base-augment",action="store_true")
    ap.add_argument("--stride-probs",nargs="+",default=["3:0.15","4:0.40","5:0.35","6:0.10"])
    ap.add_argument("--val-strides",type=int,nargs="+",default=[3,4,5,6])
    ap.add_argument("--speed-prob",type=float,default=.70);ap.add_argument("--speed-min",type=float,default=.85)
    ap.add_argument("--speed-max",type=float,default=1.18);ap.add_argument("--jitter",type=int,default=3)
    a=ap.parse_args();set_seed(a.seed);a.out_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");pr=parse_probs(a.stride_probs)

    tr=GroupDS(a.train_groups,True,pr,seed=a.seed,speed_prob=a.speed_prob,speed_min=a.speed_min,speed_max=a.speed_max,jitter=a.jitter,base_aug=not a.no_base_augment)
    train_eval=GroupDS(a.train_groups,False,pr,eval_stride="4",seed=a.seed+1,base_aug=False)
    tl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,pin_memory=torch.cuda.is_available(),worker_init_fn=worker_seed)
    tel=DataLoader(train_eval,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers)
    val_loaders={s:DataLoader(GroupDS(a.val_groups,False,pr,eval_stride=str(s),seed=a.seed+100+s,base_aug=False),batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers) for s in a.val_strides}

    model=build_model(a.num_classes).to(device);crit=nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    best={"mean_macro_f1":-1.,"epoch":-1};hist=[];best_path=a.out_dir/"gloss_translator_multistride_best.pth"

    for ep in range(1,a.epochs+1):
        model.train();tot=cor=0;ls=0.
        for x,y in tl:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
            z=logits(model(x));L=crit(z,y);L.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            p=z.argmax(1);tot+=y.numel();cor+=(p==y).sum().item();ls+=L.item()*y.numel()
        sch.step()
        if a.bn_recalibrate:recal_bn(model,tel,device)
        vals={str(s):evaluate(model,dl,device,a.num_classes) for s,dl in val_loaders.items()}
        mean_acc=float(np.mean([v["acc"] for v in vals.values()]))
        mean_f1=float(np.mean([v["macro_f1"] for v in vals.values()]))
        rec={"epoch":ep,"train_loss":ls/max(1,tot),"train_acc":cor/max(1,tot),"val_mean_acc":mean_acc,"val_mean_macro_f1":mean_f1,
             "val_by_stride":{k:{"acc":v["acc"],"macro_f1":v["macro_f1"]} for k,v in vals.items()},"lr":sch.get_last_lr()[0]}
        hist.append(rec);print(json.dumps(rec,ensure_ascii=False))
        if mean_f1>best["mean_macro_f1"]:
            best={"mean_macro_f1":mean_f1,"mean_acc":mean_acc,"epoch":ep}
            torch.save({"model":model.state_dict(),"num_classes":a.num_classes,"input_shape":[1,4,54,64],"best":best,"model_name":"FinalGlossTranslatorNet",
                        "stride_probs":pr,"speed":[a.speed_min,a.speed_max,a.speed_prob],"jitter":a.jitter},best_path)
            print("saved_best",best_path,best)
    (a.out_dir/"training_summary.json").write_text(json.dumps({"best":best,"history":hist},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
