# -*- coding: utf-8 -*-
"""
test_integrated_full.py • v12  (2025-05-12)
===========================================
- S-Rec  &  VideoVAE  두 계열 지표 동시 계산
- Moving-MNIST(64×64)  →  MS-SSIM 은 해상도 <161 일 때 NaN 처리
- 샘플 인덱스: 고정된 인덱스 기반으로 각 모델별 해당하는 잔상 데이터 사용
  ⇒ 동일한 인덱스의 원본 비디오에 대한 서로 다른 잔상 형태 사용
"""

from __future__ import annotations
import os, json, argparse, datetime, math, random
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import imageio
from PIL import Image, ImageDraw
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk

# ───── Optional deps ──────────────────────────────────────────────────────────
try:
    from piq import multi_scale_ssim as ms_ssim_piq
except Exception:
    ms_ssim_piq = None

try:
    import lpips
    lpips_alex = lpips.LPIPS(net="alex").eval()
except Exception:
    lpips_alex = None
# -----------------------------------------------------------------------------

from models import AfterimageVAE_IntegratedTraining
from utils  import get_data_loaders

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)  # PyTorch 시드 추가

# ─────────────── CLI ──────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser("Afterimage-VAE evaluation & visualisation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data_dir", default="./datasets")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--model_paths", nargs="+", required=True)
    ap.add_argument("--results_dir", default="./test_results")
    ap.add_argument("--dataset_split", choices=["train","val","test"], default="test")
    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--gpu_id", type=int, default=0)
    # 새로운 인자 추가
    ap.add_argument("--specific_samples", nargs="+", type=int, 
                    help="특정 샘플 인덱스를 직접 지정 (기본값: None)")
    return ap.parse_args()

# ─────────────── I/O ──────────────────────────────────────────────────────────
def load_model(ckpt: str, device: torch.device):
    cfg_p = os.path.join(os.path.dirname(ckpt), "config.json")
    cfg   = json.load(open(cfg_p)) if os.path.isfile(cfg_p) else {}
    model = AfterimageVAE_IntegratedTraining(
        in_channels   = cfg.get("in_channels", 1),
        latent_dim    = cfg.get("latent_dim" , 8),
        base_channels = cfg.get("base_channels", 32),
        num_frames    = cfg.get("num_frames" , 20),
        resolution    = tuple(cfg.get("resolution", [64,64])),
        device=device,
    )
    model.load(ckpt); model.to(device).eval()
    return model, cfg

def get_loader(root, ds_name, split, bs, nw):
    after = os.path.join(root,"MovingMNIST",ds_name)
    video = os.path.join(root,"MovingMNIST","mnist_test_seq.npy")
    
    tr,val,te = get_data_loaders(after, video, bs, nw)
    return {"train":tr,"val":val,"test":te}[split]

# ─────────────── metrics ──────────────────────────────────────────────────────
def mse_np(a,b): return float(np.mean((a-b)**2))

def tpsnr(p,g):
    errs=[mse_np(p[:,:,t]-p[:,:,t-1], g[:,:,t]-g[:,:,t-1])
          for t in range(1,p.shape[2])]
    if not errs: return float("nan")
    mse=np.mean(errs)
    return float("inf") if mse==0 else 10*math.log10(1/mse)

def safe_ms(pt,gt,H,W):
    if ms_ssim_piq is None or H<161 or W<161: return float("nan")
    with torch.no_grad():
        v=ms_ssim_piq(pt,gt,data_range=1.).item()
    return v if np.isfinite(v) else float("nan")

def safe_lp(p3,g3):
    if lpips_alex is None: return float("nan")
    with torch.no_grad():
        return lpips_alex(p3*2-1, g3*2-1).item()

def compute_metrics(pred: torch.Tensor, gt: torch.Tensor)->Dict[str,float]:
    p,g=pred.cpu().float(), gt.cpu().float()
    B,C,T,H,W = p.shape
    acc=dict(MSE=0,PSNR=0,SSIM=0,MS=0,LP=0,tPSNR=0)
    for b in range(B):
        acc["tPSNR"] += tpsnr(p[b].numpy(), g[b].numpy())
        for t in range(T):
            pi,gi = p[b,0,t].numpy(), g[b,0,t].numpy()
            acc["MSE"]  += mse_np(pi,gi)
            acc["PSNR"] += psnr_sk(gi,pi,data_range=1.)
            acc["SSIM"] += ssim_sk(gi,pi,data_range=1.)
            pt,gt_ = p[b,:,t:t+1].clamp(0,1), g[b,:,t:t+1].clamp(0,1)
            acc["MS"] += safe_ms(pt,gt_,H,W)
            p3,g3 = pt.repeat(1,3,1,1), gt_.repeat(1,3,1,1)
            acc["LP"] += safe_lp(p3,g3)
    frames=B*T; vids=B
    return {"MSE":acc["MSE"]/frames,
            "PSNR":acc["PSNR"]/frames,
            "SSIM":acc["SSIM"]/frames,
            "MS-SSIM":acc["MS"]/frames if ms_ssim_piq else np.nan,
            "LPIPS-Alex":acc["LP"]/frames if lpips_alex else np.nan,
            "t-PSNR":acc["tPSNR"]/vids}

# ─────────────── visuals ──────────────────────────────────────────────────────
def gif5(a,orig,vae,slot,srec,path,fps):
    C,T,H,W=orig.shape; caps=["After","Orig","V-VAE","Slot","S-Rec"]; fr=[]
    for t in range(T):
        canv=Image.new("L",(W*5,H+18),255); d=ImageDraw.Draw(canv)
        for i,img in enumerate([a[0],orig[0,t],vae[0,t],slot[0,t],srec[0,t]]):
            canv.paste(Image.fromarray((img*255).astype(np.uint8)), (W*i,18))
            d.text((W*i+W//2-18,2),caps[i],fill=0)
        d.text((6,H+2),f"F {t+1}/{T}",fill=0); fr.append(np.array(canv))
    imageio.mimsave(path,fr,fps=fps,loop=0)

def save_strip(rows:List[np.ndarray], out:str, border:int=2):
    R=len(rows); C,T,H,W=rows[0].shape
    canv=Image.new("L",((W+border)*T+border,(H+border)*R+border),255)
    for r,row in enumerate(rows):
        for t in range(T):
            img=Image.fromarray((row[0,t]*255).astype(np.uint8))
            x=border+t*(W+border); y=border+r*(H+border)
            canv.paste(img,(x,y))
    canv.save(out)

def bar(df, metric, out, high=True):
    plt.figure(figsize=(10,5))
    x=np.arange(len(df)); plt.bar(x,df[metric],0.6)
    plt.xticks(x,df["Model"],rotation=15)
    plt.ylabel(metric+(" ↑" if high else " ↓"))
    plt.title(metric+" (test)"); plt.grid(axis="y",ls="--",alpha=.4)
    plt.tight_layout(); plt.savefig(os.path.join(out,f"{metric}.png")); plt.close()

# ─────────────── main ─────────────────────────────────────────────────────────
def main():
    args=parse_args()
    dev=torch.device(f"cuda:{args.gpu_id}"
                     if args.device=="cuda" and torch.cuda.is_available()
                     else "cpu")
    root=os.path.join(args.results_dir,
                      f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(root,exist_ok=True)
    
    # 전역 시드 설정
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 모델별 설정과 데이터셋 로드
    models_and_dataloaders = []
    
    for ckpt in args.model_paths:
        model, cfg = load_model(ckpt, dev)
        
        # 모델 태그 생성
        tag = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))\
               .replace("AfterimageVAE_IntegratedTraining_","")
        
        # 데이터셋 이름 가져오기
        ds_name = cfg.get("afterimage_dataset", "mnist_afterimages.npy")
        
        # 데이터로더 생성
        loader = get_loader(
            args.data_dir,
            ds_name,
            args.dataset_split,
            args.batch_size,
            args.num_workers
        )
        
        models_and_dataloaders.append((model, loader, tag))
    
    # 첫 번째 모델의 데이터셋 길이로 샘플 인덱스 결정
    first_dataset = models_and_dataloaders[0][1].dataset
    
    # 샘플 인덱스 선택
    if args.specific_samples:
        sample_idx = args.specific_samples
    else:
        video_len = len(first_dataset)
        sample_idx = np.linspace(0, video_len-1, args.num_samples, dtype=int).tolist()
    
    print(f"Selected sample indices: {sample_idx}")
    
    # 각 모델과 데이터셋에서 동일한 인덱스에 대한 샘플 가져오기
    model_samples = []
    
    for model, loader, tag in models_and_dataloaders:
        dataset = loader.dataset
        
        # 동일한 인덱스의 샘플 가져오기 (wrap-around 처리)
        samples = []
        for idx in sample_idx:
            actual_idx = idx % len(dataset)
            samples.append(dataset[actual_idx])
        
        # 샘플 데이터를 텐서로 변환
        afterimages = torch.stack([s["afterimage"] for s in samples]).to(dev)
        videos = torch.stack([s["video"] for s in samples]).to(dev)
        
        model_samples.append((model, afterimages, videos, tag))
    
    # 메트릭스 계산과 시각화 생성
    summary = []
    
    # 모든 모델 비교를 위한 스트립 초기화
    # 원본 비디오만 있는 행은 없고, 각 샘플별로 모든 모델의 결과를 담은 행 생성
    strip_rows = []
    
    # 각 샘플 인덱스별로 모든 모델의 결과를 시각화
    for sample_num in range(len(sample_idx)):
        # 이 샘플에 대한 원본 비디오 스트립 행 (모든 모델의 원본 비디오가 동일하므로 첫 번째 모델 사용)
        strip_row = [model_samples[0][2][sample_num].cpu().numpy()]
        strip_rows.append(strip_row)
    
    # 각 모델별 처리
    for i, (model, afterimages, videos, tag) in enumerate(model_samples):
        print(f"\n▶ {tag}")
        
        # 전체 데이터셋에 대한 메트릭스 계산
        loader = models_and_dataloaders[i][1]
        total_s, total_v, n = {}, {}, 0
        vae_key = None
        
        for k in ["video_recon", "vae_recon", "video_vae", "vae_video",
                 "video_recon_video", "recon_video"]:
            if hasattr(model, "last_out") or True:
                vae_key = k
                break
        
        for batch in tqdm(loader, desc="Eval", leave=False):
            a, v = batch["afterimage"].to(dev), batch["video"].to(dev)
            with torch.no_grad(): out = model(a, v)
            m_s = compute_metrics(out["after_recon"], v)
            m_v = compute_metrics(out[vae_key], v)
            for k, v_ in m_s.items(): total_s[k] = total_s.get(k, 0) + v_
            for k, v_ in m_v.items(): total_v[k] = total_v.get(k, 0) + v_
            n += 1
        
        S = {f"S-{k}": v/n for k, v in total_s.items()}
        V = {f"V-{k}": v/n for k, v in total_v.items()}
        summary.append({"Model": tag, **S, **V})
        
        s_line = " | ".join(f"{k}:{v:.4f}" for k, v in S.items() if k.startswith("S-PSNR") or k.startswith("S-SSIM"))
        v_line = " | ".join(f"{k}:{v:.4f}" for k, v in V.items() if k.startswith("V-PSNR") or k.startswith("V-SSIM"))
        print("  S-Rec  ", s_line)
        print("  V-VAE  ", v_line)
        
        # 모델별 샘플 처리
        vdir = os.path.join(root, tag)
        os.makedirs(vdir, exist_ok=True)
        
        # 시각화용 샘플 처리
        with torch.no_grad():
            out = model(afterimages, videos)
        
        vae = out[vae_key]
        slot = out["slot_video"]
        srec = out["after_recon"]
        
        # 각 샘플에 대한 GIF 생성 및 스트립 업데이트
        for j, (a0, v0, va, sr, sl) in enumerate(zip(afterimages, videos, vae, srec, slot)):
            # 각 모델의 결과를 스트립에 추가
            strip_rows[j].append(sr.cpu().numpy())
            
            # GIF 생성
            gif5(a0.cpu().numpy(), v0.cpu().numpy(), va.cpu().numpy(),
                 sl.cpu().numpy(), sr.cpu().numpy(),
                 os.path.join(vdir, f"sample{j+1}.gif"), args.fps)
    
    # 각 샘플별 모든 모델의 결과를 담은 스트립 생성
    for i, r in enumerate(strip_rows, 1):
        save_strip(r, os.path.join(root, f"sample{i}_strip.png"))
    
    # 요약 테이블 및 차트 생성
    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(root, "metrics_summary.csv"), index=False)
    df.to_json(os.path.join(root, "metrics_summary.json"), orient="records", indent=2)
    
    bar(df.rename(columns={"S-PSNR": "PSNR"}), "PSNR", root, True)
    bar(df.rename(columns={"S-MS-SSIM": "MS-SSIM"}), "MS-SSIM", root, True)
    bar(df.rename(columns={"S-LPIPS-Alex": "LPIPS-Alex"}), "LPIPS-Alex", root, False)
    
    print(f"\n✔︎ Results saved → {root}")

if __name__ == "__main__":
    main()