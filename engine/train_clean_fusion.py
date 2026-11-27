from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from datasets.registry import load_dataset
from datasets.common import make_loaders, seed_everything
from datasets.spec import DatasetSpec
from models.clean_fusion import CleanFusionBackbone, count_parameters
from random_control import derived_seed
from utils.config import load_experiment
from utils.pca import fit_training_pca
from utils.data_signature import signature
from utils.resume import ResumeStore, capture_random, restore_random, atomic_rows, atomic_torch


def parse_args():
    p = argparse.ArgumentParser(description="Train PCRFusion Stage-I Clean Fusion Learning with fixed PCA")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--split-file", type=Path, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--model-seed", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume from the last complete epoch")
    return p.parse_args()


def class_weights(loader, classes: int, device: torch.device) -> Tensor:
    counts = torch.zeros(classes, dtype=torch.float64)
    for b in loader:
        counts += torch.bincount(b["label"], minlength=classes).double()
    w = counts.sum() / counts.clamp_min(1.0)
    return (w / w.mean()).float().to(device)


def joint_spatial_augment(h: Tensor, l: Tensor) -> Tuple[Tensor, Tensor]:
    k = int(torch.randint(0, 4, (1,), device=h.device).item())
    if k:
        h = torch.rot90(h, k, dims=(-2, -1)); l = torch.rot90(l, k, dims=(-2, -1))
    if bool(torch.rand((), device=h.device) < 0.5):
        h = torch.flip(h, dims=(-1,)); l = torch.flip(l, dims=(-1,))
    if bool(torch.rand((), device=h.device) < 0.5):
        h = torch.flip(h, dims=(-2,)); l = torch.flip(l, dims=(-2,))
    return h, l


def info_nce(z1: Tensor, z2: Tensor, temperature: float) -> Tensor:
    if z1.shape[0] <= 1:
        return z1.new_zeros(())
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    logits = (z1 @ z2.t()) / max(float(temperature), 1e-6)
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def lr_factor(epoch: int, epochs: int, warmup: int, min_ratio: float) -> float:
    e = epoch + 1
    if warmup > 0 and e <= warmup:
        return max(1e-3, e / warmup)
    progress = (e - warmup) / max(1, epochs - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_ratio + (1.0 - min_ratio) * cosine


def classification_metrics(y, p, classes: int) -> Dict[str, object]:
    y = np.asarray(y, np.int64); p = np.asarray(p, np.int64)
    cm = np.zeros((classes, classes), dtype=np.int64)
    for a, b in zip(y, p):
        if 0 <= a < classes and 0 <= b < classes:
            cm[a, b] += 1
    oa = float((y == p).mean())
    recalls = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(classes, dtype=float), where=cm.sum(1) > 0)
    valid = cm.sum(1) > 0
    aa = float(recalls[valid].mean()) if valid.any() else float("nan")
    total = cm.sum(); pe = float((cm.sum(0) * cm.sum(1)).sum()) / float(max(total * total, 1))
    kappa = float((oa - pe) / max(1.0 - pe, 1e-12))
    return {"OA": oa, "AA": aa, "Kappa": kappa, "EachAcc": recalls.tolist(), "confusion_matrix": cm.tolist()}


@torch.no_grad()
def evaluate(model, loader, device, classes: int) -> Dict[str, object]:
    model.eval(); ys=[]; pf=[]; ph=[]; pl=[]
    for b in loader:
        h=b["hsi"].to(device, non_blocking=True); l=b["lidar"].to(device, non_blocking=True)
        out=model(h,l)
        ys.append(b["label"].numpy()); pf.append(out["logits"].argmax(1).cpu().numpy())
        ph.append(out["hsi_logits"].argmax(1).cpu().numpy()); pl.append(out["lidar_logits"].argmax(1).cpu().numpy())
    y=np.concatenate(ys); result=classification_metrics(y,np.concatenate(pf),classes)
    result["HSI_OA"]=classification_metrics(y,np.concatenate(ph),classes)["OA"]
    result["LiDAR_OA"]=classification_metrics(y,np.concatenate(pl),classes)["OA"]
    result["FusionGain_vs_HSI_pp"]=(float(result["OA"])-float(result["HSI_OA"]))*100.0
    return result


def _is_better(current: Dict[str, object], best: Dict[str, object] | None) -> bool:
    if best is None: return True
    if float(current["OA"]) != float(best["OA"]): return float(current["OA"]) > float(best["OA"])
    return float(current["AA"]) > float(best["AA"])


def main():
    args=parse_args()
    bundle=load_experiment(args.config,data_root_override=args.data_root,split_file_override=args.split_file,
                           model_seed_override=args.model_seed,batch_size_override=args.batch_size,device_override=args.device)
    main_cfg=bundle["config"]; runtime=dict(main_cfg["runtime"]); cfg=dict(main_cfg["clean_fusion"])
    model_seed=int(main_cfg["seeds"]["model"]); seed_everything(model_seed)
    device_name=str(runtime.get("device","cuda")); device=torch.device(device_name if not device_name.startswith("cuda") or torch.cuda.is_available() else "cpu")
    amp=bool(args.amp or runtime.get("amp",False)); spec=DatasetSpec.from_config_bundle(bundle,args.data_root)
    dataset=load_dataset(bundle,spec.data_root); loaders=make_loaders(dataset,spec)
    if getattr(loaders["train"],"generator",None) is not None:
        loaders["train"].generator.manual_seed(derived_seed(model_seed,"clean_fusion_loader"))

    pca_cfg=dict(cfg.get("pca",{})); pca_enabled=bool(pca_cfg.get("enabled",True)); pca_bands=int(pca_cfg.get("components",20)) if pca_enabled else 0
    pca_ratio=[]
    if pca_enabled:
        mean, components, ratio=fit_training_pca(dataset.hsi,dataset.train_labels,pca_bands)
        pca_bands=int(components.shape[0]); pca_ratio=ratio.tolist()
    else:
        mean=components=None

    epochs=2 if args.smoke_test else int(cfg["epochs"])
    model=CleanFusionBackbone(dataset.num_bands,dataset.lidar_channels,dataset.num_classes,
        dim=int(cfg["feature_dim"]),heads=int(cfg["heads"]),dropout=float(cfg["dropout"]),fusion=str(cfg["fusion"]),
        mixstyle_p=float(cfg.get("mixstyle_probability",0.0)),pca_components=pca_bands).to(device)
    if pca_enabled: model.set_hsi_pca(mean,components)
    params=count_parameters(model)
    print("="*88); print("DARFusion | STAGE I: CLEAN FUSION LEARNING | CONFIGURED CHECKPOINT SELECTION")
    print(f"dataset={dataset.dataset_name} train={len(loaders['train'].dataset)} test={len(loaders['test'].dataset)} patch={spec.patch_size} batch={spec.batch_size} epochs={epochs}")
    print(f"PCA={'PCA'+str(pca_bands) if pca_enabled else 'disabled'} fitted from configured training-label pixels only")
    print(f"params={params:,} ({params/1e6:.3f} M)")
    print(f"[selection] {dataset.metadata.get('split',{}).get('evaluation_split','configured_test')} labels select Stage I.")
    print("="*88)

    weight=None if str(cfg.get("class_weighting","none"))=="none" else class_weights(loaders["train"],dataset.num_classes,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]))
    min_ratio=max(float(cfg["min_learning_rate"])/max(float(cfg["learning_rate"]),1e-12),0.0)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda ep:lr_factor(ep,epochs,int(cfg["warmup_epochs"]),min_ratio))
    amp_active=bool(amp and device.type=="cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=amp_active) if hasattr(torch,"amp") and hasattr(torch.amp,"GradScaler") else torch.cuda.amp.GradScaler(enabled=amp_active)
    use_aug=bool(cfg.get("spatial_augmentation",True)); aux_w=float(cfg.get("auxiliary_loss_weight",0.0)); con_w=float(cfg.get("contrastive_loss_weight",0.0))

    output=Path(args.output_root).expanduser().resolve(); output.mkdir(parents=True,exist_ok=True)
    store=ResumeStore(output,dict(stage='clean',config=main_cfg,data=signature(dataset,spec),epochs=epochs),args.resume)
    recovered=store.load()
    start=time.time(); elapsed_before=0.0; history=[]; best_metrics=None; best_state=None; best_epoch=None; start_epoch=1
    if recovered is not None:
        model.load_state_dict(recovered['model']);optimizer.load_state_dict(recovered['optimizer'])
        scheduler.load_state_dict(recovered['scheduler']);scaler.load_state_dict(recovered['scaler'])
        history=recovered['history'];best_metrics=recovered['best_metrics'];best_state=recovered['best_state'];best_epoch=recovered['best_epoch']
        elapsed_before=recovered['elapsed_seconds'];start_epoch=int(recovered['epoch'])+1
        restore_random(recovered['random'],loaders)
        print(f"[resume:StageI] next epoch={start_epoch}/{epochs}",flush=True)
    for epoch in range(start_epoch,epochs+1):
        model.train(); total=correct=0; loss_sum=cls_sum=con_sum=0.0
        for b in loaders["train"]:
            h=b["hsi"].to(device,non_blocking=True); l=b["lidar"].to(device,non_blocking=True); y=b["label"].to(device,non_blocking=True)
            if use_aug: h,l=joint_spatial_augment(h,l)
            optimizer.zero_grad(set_to_none=True); ctx=torch.autocast(device_type="cuda",dtype=torch.float16) if amp and device.type=="cuda" else nullcontext()
            with ctx:
                out=model(h,l); fused=F.cross_entropy(out["logits"],y,weight=weight,label_smoothing=float(cfg.get("label_smoothing",0.0)))
                if aux_w>0:
                    h_loss=F.cross_entropy(out["hsi_logits"],y,weight=weight); l_loss=F.cross_entropy(out["lidar_logits"],y,weight=weight)
                else: h_loss=l_loss=fused.new_zeros(())
                con=info_nce(out["hsi_projection"],out["lidar_projection"],float(cfg.get("contrastive_temperature",0.1))) if con_w>0 else fused.new_zeros(())
                loss=fused+aux_w*0.5*(h_loss+l_loss)+con_w*con
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),float(cfg["grad_clip"])); scaler.step(optimizer); scaler.update()
            n=y.numel(); total+=n; correct+=int((out["logits"].argmax(1)==y).sum()); loss_sum+=float(loss.detach())*n; cls_sum+=float(fused.detach())*n; con_sum+=float(con.detach())*n
        scheduler.step(); metrics=evaluate(model,loaders["test"],device,dataset.num_classes)
        if _is_better(metrics,best_metrics): best_metrics=copy.deepcopy(metrics); best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch
        row={"epoch":epoch,"loss":loss_sum/max(total,1),"fused_ce":cls_sum/max(total,1),"contrastive":con_sum/max(total,1),"train_OA":correct/max(total,1),
             "test_OA":metrics["OA"],"test_AA":metrics["AA"],"test_Kappa":metrics["Kappa"],"lr":optimizer.param_groups[0]["lr"]}; history.append(row)
        store.save(dict(epoch=epoch,goal_epochs=epochs,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                        history=history,best_metrics=best_metrics,best_state=best_state,best_epoch=best_epoch,
                        elapsed_seconds=elapsed_before+time.time()-start,random=capture_random(loaders)))
        atomic_rows(output/'history.csv',history)
        print(f"[CleanFusion] epoch={epoch:03d} loss={row['loss']:.5f} train_OA={row['train_OA']:.5f} test_OA={metrics['OA']:.5f} test_AA={metrics['AA']:.5f}{' *best' if best_epoch==epoch else ''}")

    if best_state is None or best_metrics is None: raise RuntimeError("No Clean Fusion checkpoint selected")
    output=Path(args.output_root).expanduser().resolve(); output.mkdir(parents=True,exist_ok=True)
    last_state=copy.deepcopy(model.state_dict()); model.load_state_dict(best_state)
    common_args={"dim":int(cfg["feature_dim"]),"heads":int(cfg["heads"]),"dropout":float(cfg["dropout"]),"fusion":str(cfg["fusion"]),
                 "mixstyle_p":float(cfg.get("mixstyle_probability",0.0)),"pca_components":pca_bands,"pca_enabled":pca_enabled,
                 "patch_size":spec.patch_size,"batch_size":spec.batch_size,"seed":model_seed}
    ckpt={"model":best_state,"args":common_args,"result":best_metrics,"component":"CleanFusionBackbone","selection":"best_validation_OA" if bool(main_cfg["validation"].get("enabled", False)) else "best_official_test_OA",
          "best_epoch":best_epoch,"config_hash":bundle["hashes"]["experiment_config_sha1"],"protocol_warning":("validation labels used for checkpoint selection" if bool(main_cfg["validation"].get("enabled", False)) else "official test labels used for checkpoint selection"),
          "data_signature":signature(dataset,spec), "selection_protocol":main_cfg["selection_protocol"]}
    atomic_torch(output/"best.pt",ckpt); atomic_torch(output/"last.pt",{"model":last_state,"args":common_args,"component":"CleanFusionBackbone","selection":"last_epoch"})
    result=dict(best_metrics); result.update({"dataset":dataset.dataset_name,"model_seed":model_seed,"patch_size":spec.patch_size,"batch_size":spec.batch_size,
        "epochs_ran":len(history),"best_epoch":best_epoch,"parameters":params,"training_seconds":elapsed_before+time.time()-start,"evaluation_split":dataset.metadata.get("split",{}).get("evaluation_split","configured_test"),
        "checkpoint_selection":ckpt["selection"],"pca_enabled":pca_enabled,"pca_components":pca_bands,"pca_explained_variance_ratio_sum":float(sum(pca_ratio)) if pca_ratio else None})
    (output/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    with (output/"history.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)
    print("\n"+"="*88); print("CLEAN FUSION SELECTED RESULT"); print(f"OA={float(result['OA'])*100:.3f}% AA={float(result['AA'])*100:.3f}% Kappa={float(result['Kappa']):.5f} best_epoch={best_epoch}"); print(f"saved={output/'best.pt'}"); print("="*88)


if __name__=="__main__": main()
