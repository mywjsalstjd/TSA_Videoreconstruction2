# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from PIL import Image, ImageDraw
import imageio
import matplotlib.pyplot as plt

def save_comparison_gif_pretrained(model, data_loader, device='cuda', out_path="comparison.gif", num_samples=5, fps=5):
    """
    AfterimageVAE_PretrainedVAE 모델의 결과를 GIF로 저장합니다.
    
    Args:
        model: 평가할 모델
        data_loader: 데이터 로더
        device: 평가 장치
        out_path: GIF 저장 경로
        num_samples: 표시할 샘플 수
        fps: GIF 프레임 속도
    """
    model.eval()
    
    # 샘플 데이터 얻기
    for batch in data_loader:
        afterimage = batch['afterimage'][:num_samples].to(device)
        video = batch['video'][:num_samples].to(device)
        break
    
    B = afterimage.shape[0]  # 배치 크기
    T = video.shape[2]       # 프레임 수
    
    with torch.no_grad():
        # 잔상 이미지에서 비디오 생성
        after_recon, _, _, slot_video, attention_masks = model(afterimage)
    
    # NumPy 배열로 변환
    video_np = video.cpu().numpy()
    after_recon_np = after_recon.cpu().numpy()
    slot_video_np = slot_video.cpu().numpy()
    afterimage_np = afterimage.cpu().numpy()
    attention_masks_np = attention_masks.cpu().numpy()
    
    # GIF 프레임 생성
    frames_list = []
    
    for t in range(T):
        # 모든 샘플에 대한 모자이크 이미지 생성
        # 각 행: [잔상 이미지] [원본 프레임] [슬롯 생성] [최종 재구성] [어텐션 마스크]
        mosaic_w = 64 * 5  # 5개 열
        mosaic_h = 64 * B  # B개 행 (샘플 수)
        
        big_frame = Image.new('L', (mosaic_w, mosaic_h))
        
        for i in range(B):
            # 잔상 이미지 (모든 프레임에서 동일)
            after_img = (afterimage_np[i, 0] * 255).clip(0, 255).astype(np.uint8)
            after_img = Image.fromarray(after_img, 'L')
            
            # 원본 비디오 프레임
            orig_frame = (video_np[i, 0, t] * 255).clip(0, 255).astype(np.uint8)
            orig_img = Image.fromarray(orig_frame, 'L')
            
            # 슬롯에서 직접 생성된 프레임
            slot_frame = (slot_video_np[i, 0, t] * 255).clip(0, 255).astype(np.uint8)
            slot_img = Image.fromarray(slot_frame, 'L')
            
            # 잔상에서 재구성 프레임
            after_recon_frame = (after_recon_np[i, 0, t] * 255).clip(0, 255).astype(np.uint8)
            after_recon_img = Image.fromarray(after_recon_frame, 'L')
            
            # 어텐션 마스크
            attn_mask = (attention_masks_np[i, 0, t] * 255).clip(0, 255).astype(np.uint8)
            attn_img = Image.fromarray(attn_mask, 'L')
            
            # 이미지 배치
            y_offset = i * 64
            big_frame.paste(after_img, (0, y_offset))
            big_frame.paste(orig_img, (64, y_offset))
            big_frame.paste(slot_img, (128, y_offset))
            big_frame.paste(after_recon_img, (192, y_offset))
            big_frame.paste(attn_img, (256, y_offset))
        
        # 프레임에 레이블 추가
        if t == 0:
            # 첫 프레임에만 레이블 추가
            draw = ImageDraw.Draw(big_frame)
            labels = ["Afterimage", "Original", "Slot Generated", "Final Output", "Attention Mask"]
            for idx, label in enumerate(labels):
                draw.text((idx * 64 + 5, 5), label, fill=255)
            
            # 프레임 정보 추가
            draw.text((mosaic_w - 100, mosaic_h - 15), f"Frame: {t}", fill=255)
        else:
            # 다른 프레임에도 프레임 정보 추가
            draw = ImageDraw.Draw(big_frame)
            draw.text((mosaic_w - 100, mosaic_h - 15), f"Frame: {t}", fill=255)
        
        frames_list.append(np.array(big_frame))
    
    # GIF 저장
    imageio.mimsave(out_path, frames_list, fps=fps, loop=0)
    print(f"저장된 GIF => {out_path}")

def save_comparison_gif_integrated(model, data_loader, device='cuda', out_path="comparison.gif", num_samples=5, fps=5):
    """
    AfterimageVAE_IntegratedTraining 모델의 결과를 GIF로 저장합니다.
    
    Args:
        model: 평가할 모델
        data_loader: 데이터 로더
        device: 평가 장치
        out_path: GIF 저장 경로
        num_samples: 표시할 샘플 수
        fps: GIF 프레임 속도
    """
    model.eval()
    
    # 샘플 데이터 얻기
    for batch in data_loader:
        afterimage = batch['afterimage'][:num_samples].to(device)
        video = batch['video'][:num_samples].to(device)
        break
    
    B = afterimage.shape[0]  # 배치 크기
    T = video.shape[2]       # 프레임 수
    
    with torch.no_grad():
        # 모델 출력 얻기
        outputs = model(afterimage, video)
        slot_video = outputs['slot_video']
        after_recon = outputs['after_recon']
        video_recon = outputs['video_recon']
        attention_masks = outputs['attention_masks']
    
    # NumPy 배열로 변환
    video_np = video.cpu().numpy()
    slot_video_np = slot_video.cpu().numpy()
    after_recon_np = after_recon.cpu().numpy()
    video_recon_np = video_recon.cpu().numpy()
    afterimage_np = afterimage.cpu().numpy()
    attention_masks_np = attention_masks.cpu().numpy()
    
    # GIF 프레임 생성
    frames_list = []
    
    for t in range(T):
        # 모든 샘플에 대한 모자이크 이미지 생성
        # 각 행: [잔상 이미지] [원본 프레임] [슬롯 생성] [최종 출력] [VideoVAE 출력] [어텐션 마스크]
        mosaic_w = 64 * 6  # 6개 열
        mosaic_h = 64 * B  # B개 행 (샘플 수)
        
        big_frame = Image.new('L', (mosaic_w, mosaic_h))
        
        for i in range(B):
            # 잔상 이미지 (모든 프레임에서 동일)
            after_img = (afterimage_np[i, 0] * 255).clip(0, 255).astype(np.uint8)
            after_img = Image.fromarray(after_img, 'L')
            
            # 원본 비디오 프레임
            orig_frame = (video_np[i, 0, t] * 255).clip(0, 255).astype(np.uint8)
            orig_img = Image.fromarray(orig_frame, 'L')
            
            # 슬롯에서 직접 생성된 프레임
            slot_frame = (slot_video_np[i, 0, t] * 255).clip(0, 255).astype(np.uint8)
            slot_img = Image.fromarray(slot_frame, 'L')
            
            # 최종 출력 프레임
            after_recon_frame = (after_recon_np[i, 0, t] * 255).clip(0, 255).astype(np.uint8)
            after_recon_img = Image.fromarray(after_recon_frame, 'L')
            
            # VideoVAE 재구성 프레임
            vae_recon_frame = (video_recon_np[i, 0, t] * 255).clip(0, 255).astype(np.uint8)
            vae_recon_img = Image.fromarray(vae_recon_frame, 'L')
            
            # 어텐션 마스크
            attn_mask = (attention_masks_np[i, 0, t] * 255).clip(0, 255).astype(np.uint8)
            attn_img = Image.fromarray(attn_mask, 'L')
            
            # 이미지 배치
            y_offset = i * 64
            big_frame.paste(after_img, (0, y_offset))
            big_frame.paste(orig_img, (64, y_offset))
            big_frame.paste(slot_img, (128, y_offset))
            big_frame.paste(after_recon_img, (192, y_offset))
            big_frame.paste(vae_recon_img, (256, y_offset))
            big_frame.paste(attn_img, (320, y_offset))
        
        # 프레임에 레이블 추가
        if t == 0:
            # 첫 프레임에만 레이블 추가
            draw = ImageDraw.Draw(big_frame)
            labels = ["Afterimage", "Original", "Slot Video", "Final Output", "VideoVAE", "Attention Mask"]
            for idx, label in enumerate(labels):
                draw.text((idx * 64 + 5, 5), label, fill=255)
            
            # 프레임 정보 추가
            draw.text((mosaic_w - 100, mosaic_h - 15), f"Frame: {t}", fill=255)
        else:
            # 다른 프레임에도 프레임 정보 추가
            draw = ImageDraw.Draw(big_frame)
            draw.text((mosaic_w - 100, mosaic_h - 15), f"Frame: {t}", fill=255)
        
        frames_list.append(np.array(big_frame))
    
    # GIF 저장
    imageio.mimsave(out_path, frames_list, fps=fps, loop=0)
    print(f"저장된 GIF => {out_path}")

def plot_training_curves_pretrained(history, save_dir):
    """
    AfterimageVAE_PretrainedVAE 모델의 학습 과정을 그래프로 시각화합니다.
    
    Args:
        history (dict): 학습 히스토리 딕셔너리
        save_dir (str): 그래프를 저장할 디렉토리
    """
    # 학습 손실 그래프
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Pretrained VAE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'train_loss.png'))
    plt.close()
    
    # 개별 손실 그래프
    losses = ['z1', 'z2', 'video', 'temporal']
    plt.figure(figsize=(12, 8))
    for loss_name in losses:
        plt.plot([h[loss_name] for h in history['train_loss_components']], label=f'{loss_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Components (Pretrained VAE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_components.png'))
    plt.close()

    # 검증 지표 그래프
    metrics_to_plot = ['MSE', 'PSNR', 'SSIM', 'temporal_consistency']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 5))
        vals = [v[metric] for v in history['val_metrics']]
        plt.plot(vals, 'r-', label=f'{metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'Validation {metric} (Pretrained VAE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'val_{metric}.png'))
        plt.close()

def plot_training_curves_integrated(history, save_dir):
    """
    AfterimageVAE_IntegratedTraining 모델의 학습 과정을 그래프로 시각화합니다.
    
    Args:
        history (dict): 학습 히스토리 딕셔너리
        save_dir (str): 그래프를 저장할 디렉토리
    """
    # 학습 손실 그래프
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Integrated Training)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'train_loss.png'))
    plt.close()
    
    # 개별 손실 그래프
    plt.figure(figsize=(14, 8))
    losses = ['z1', 'z2', 'slot_video', 'after_recon', 'video_recon', 'temporal']
    for loss_name in losses:
        plt.plot([h[loss_name] for h in history['train_loss_components']], label=f'{loss_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Components (Integrated Training)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_components.png'))
    plt.close()

    # 검증 지표 그래프 - MSE
    plt.figure(figsize=(12, 6))
    plt.plot([v['slot']['MSE'] for v in history['val_metrics']], 'b-', label='Slot Video')
    plt.plot([v['final']['MSE'] for v in history['val_metrics']], 'r-', label='Final Output')
    plt.plot([v['vae']['MSE'] for v in history['val_metrics']], 'g-', label='VideoVAE Output')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Validation MSE (Integrated Training)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'val_mse.png'))
    plt.close()
    
    # 검증 지표 그래프 - PSNR
    plt.figure(figsize=(12, 6))
    plt.plot([v['slot']['PSNR'] for v in history['val_metrics']], 'b-', label='Slot Video')
    plt.plot([v['final']['PSNR'] for v in history['val_metrics']], 'r-', label='Final Output')
    plt.plot([v['vae']['PSNR'] for v in history['val_metrics']], 'g-', label='VideoVAE Output')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Validation PSNR (Integrated Training)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'val_psnr.png'))
    plt.close()
    
    # 검증 지표 그래프 - SSIM
    plt.figure(figsize=(12, 6))
    plt.plot([v['slot']['SSIM'] for v in history['val_metrics']], 'b-', label='Slot Video')
    plt.plot([v['final']['SSIM'] for v in history['val_metrics']], 'r-', label='Final Output')
    plt.plot([v['vae']['SSIM'] for v in history['val_metrics']], 'g-', label='VideoVAE Output')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Validation SSIM (Integrated Training)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'val_ssim.png'))
    plt.close()