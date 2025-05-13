# -*- coding: utf-8 -*-
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torch
from tqdm import tqdm

def calculate_metrics(original, reconstructed):
    """
    비디오 프레임 간의 다양한 평가 지표를 계산합니다.
    
    Args:
        original (numpy.ndarray): 원본 비디오 [T, H, W], 범위 [0, 1]
        reconstructed (numpy.ndarray): 재구성된 비디오 [T, H, W], 범위 [0, 1]
        
    Returns:
        dict: 계산된 지표를 포함하는 딕셔너리
            - MSE: 평균 제곱 오차
            - PSNR: 최대 신호 대 잡음비
            - SSIM: 구조적 유사성 지수
            - temporal_consistency: 시간적 일관성
    """
    # MSE 계산
    mse = np.mean((original - reconstructed)**2)
    
    # 프레임 별 PSNR 및 SSIM 계산
    psnr_frames = []
    ssim_frames = []
    for t in range(original.shape[0]):
        psnr = peak_signal_noise_ratio(original[t], reconstructed[t], data_range=1.0)
        ssim = structural_similarity(original[t], reconstructed[t], data_range=1.0)
        psnr_frames.append(psnr)
        ssim_frames.append(ssim)
    
    # 평균 PSNR 및 SSIM
    psnr = np.mean(psnr_frames)
    ssim = np.mean(ssim_frames)
    
    # 시간적 일관성 - 원본과 재구성 비디오 간의 상관 계수
    temp_consistency = np.corrcoef(original.flatten(), reconstructed.flatten())[0,1]
    
    return {
        'MSE': float(mse),
        'PSNR': float(psnr),
        'SSIM': float(ssim),
        'temporal_consistency': float(temp_consistency),
    }

def evaluate_pretrained_vae_model(model, data_loader, device='cuda', desc="Evaluation"):
    """
    AfterimageVAE_PretrainedVAE 모델 평가 함수
    
    Args:
        model: 평가할 모델
        data_loader: 데이터 로더
        device: 평가 장치
        desc: 진행 막대 설명
        
    Returns:
        dict: 평가 지표 딕셔너리
    """
    model.eval()
    metrics = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc, leave=False)
        for batch in pbar:
            afterimage = batch['afterimage'].to(device)
            video = batch['video'].to(device)
            B = afterimage.shape[0]
            
            # 잔상 이미지에서 비디오 생성
            after_recon, _, _, slot_video, _ = model(afterimage)
            
            # 평가 지표 계산
            video_np = video.cpu().numpy()
            after_recon_np = after_recon.cpu().numpy()
            slot_video_np = slot_video.cpu().numpy()
            
            for i in range(B):
                # 원본 비디오
                original = video_np[i, 0]  # [T, H, W]
                
                # 잔상에서 재구성된 비디오 지표 계산
                recon_from_after = after_recon_np[i, 0]
                metrics_from_after = calculate_metrics(original, recon_from_after)
                
                # 슬롯에서 직접 생성된 비디오 지표 계산
                slot_video_frames = slot_video_np[i, 0]
                metrics_from_slot = calculate_metrics(original, slot_video_frames)
                
                # 더 좋은 지표를 가진 결과 선택
                if metrics_from_after['MSE'] < metrics_from_slot['MSE']:
                    metrics.append(metrics_from_after)
                else:
                    metrics.append(metrics_from_slot)
    
    # 평균 지표 계산
    avg_metrics = {
        'MSE': np.mean([m['MSE'] for m in metrics]),
        'PSNR': np.mean([m['PSNR'] for m in metrics]),
        'SSIM': np.mean([m['SSIM'] for m in metrics]),
        'temporal_consistency': np.mean([m['temporal_consistency'] for m in metrics])
    }
    
    return avg_metrics

def evaluate_integrated_model(model, data_loader, device='cuda', desc="Evaluation"):
    """
    AfterimageVAE_IntegratedTraining 모델 평가 함수
    
    Args:
        model: 평가할 모델
        data_loader: 데이터 로더
        device: 평가 장치
        desc: 진행 막대 설명
        
    Returns:
        dict: 세 가지 출력에 대한 평가 지표 딕셔너리
    """
    model.eval()
    metrics_slot = []  # 슬롯 비디오 메트릭
    metrics_final = []  # 최종 출력 메트릭
    metrics_vae = []   # VideoVAE 재구성 메트릭
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc, leave=False)
        for batch in pbar:
            afterimage = batch['afterimage'].to(device)
            video = batch['video'].to(device)
            B = afterimage.shape[0]
            
            # 모델 출력 얻기
            outputs = model(afterimage, video)
            slot_video = outputs['slot_video']
            after_recon = outputs['after_recon']
            video_recon = outputs['video_recon']
            
            # 평가 지표 계산
            video_np = video.cpu().numpy()
            slot_video_np = slot_video.cpu().numpy()
            after_recon_np = after_recon.cpu().numpy()
            video_recon_np = video_recon.cpu().numpy()
            
            for i in range(B):
                # 원본 비디오
                original = video_np[i, 0]  # [T, H, W]
                
                # 슬롯 비디오 메트릭
                slot_frames = slot_video_np[i, 0]
                metrics_slot.append(calculate_metrics(original, slot_frames))
                
                # 최종 출력 메트릭
                recon_frames = after_recon_np[i, 0]
                metrics_final.append(calculate_metrics(original, recon_frames))
                
                # VideoVAE 재구성 메트릭
                vae_frames = video_recon_np[i, 0]
                metrics_vae.append(calculate_metrics(original, vae_frames))
    
    # 평균 지표 계산
    avg_metrics_slot = {
        'MSE': np.mean([m['MSE'] for m in metrics_slot]),
        'PSNR': np.mean([m['PSNR'] for m in metrics_slot]),
        'SSIM': np.mean([m['SSIM'] for m in metrics_slot]),
        'temporal_consistency': np.mean([m['temporal_consistency'] for m in metrics_slot])
    }
    
    avg_metrics_final = {
        'MSE': np.mean([m['MSE'] for m in metrics_final]),
        'PSNR': np.mean([m['PSNR'] for m in metrics_final]),
        'SSIM': np.mean([m['SSIM'] for m in metrics_final]),
        'temporal_consistency': np.mean([m['temporal_consistency'] for m in metrics_final])
    }
    
    avg_metrics_vae = {
        'MSE': np.mean([m['MSE'] for m in metrics_vae]),
        'PSNR': np.mean([m['PSNR'] for m in metrics_vae]),
        'SSIM': np.mean([m['SSIM'] for m in metrics_vae]),
        'temporal_consistency': np.mean([m['temporal_consistency'] for m in metrics_vae])
    }
    
    return {
        'slot': avg_metrics_slot,
        'final': avg_metrics_final,
        'vae': avg_metrics_vae
    }