# -*- coding: utf-8 -*-
import os
import datetime
import logging
import json
import torch
import torch.nn.functional as F

def setup_logging(checkpoint_dir):
    """
    로깅 설정을 초기화합니다.
    
    Args:
        checkpoint_dir (str): 로그 파일을 저장할 디렉토리
        
    Returns:
        logging.Logger: 설정된 로거
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(checkpoint_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_checkpoint_dir(base_dir, model_name):
    """
    체크포인트 디렉토리를 생성합니다.
    
    Args:
        base_dir (str): 기본 체크포인트 디렉토리
        model_name (str): 모델 이름
        
    Returns:
        str: 생성된 체크포인트 디렉토리 경로
    """
    # 먼저 모델별 디렉토리를 확인/생성
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # 그 안에 타임스탬프 디렉토리 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(model_dir, timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return checkpoint_dir

def create_results_dir(base_dir, model_name, timestamp=None):
    """
    결과 디렉토리를 생성합니다.
    
    Args:
        base_dir (str): 기본 결과 디렉토리
        model_name (str): 모델 이름
        timestamp (str, optional): 타임스탬프 (없으면 현재 시간 사용)
        
    Returns:
        str: 생성된 결과 디렉토리 경로
    """
    # 먼저 모델별 디렉토리를 확인/생성
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # 타임스탬프 디렉토리 생성
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(model_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    return results_dir

def compute_loss_pretrained_vae(model, afterimage, target_video, alpha=1.0, beta=1.0, gamma=0.2, delta=0.05):
    """
    AfterimageVAE_PretrainedVAE 모델의 손실 계산
    
    Args:
        model: AfterimageVAE_PretrainedVAE 모델
        afterimage (torch.Tensor): 잔상 이미지 [B, C, H, W]
        target_video (torch.Tensor): 원본 비디오 [B, C, T, H, W]
        alpha (float): z1 손실 가중치
        beta (float): z2 손실 가중치
        gamma (float): 비디오 재구성 손실 가중치
        delta (float): 시간적 일관성 손실 가중치
        
    Returns:
        tuple:
            - total_loss (torch.Tensor): 총 손실
            - loss_dict (dict): 개별 손실 항목 사전
    """
    # 모델 순전파
    x_recon, z1_after, z2_after, video_from_afterimage, _ = model(afterimage)
    
    # 원본 비디오에서 참조 잠재 표현 추출
    z1_video, z2_video = model.get_reference_latents(target_video)
    
    # 1. z1 정렬 손실 - 확장된 비디오의 z1이 원본 비디오의 z1과 일치하도록
    loss_z1 = F.mse_loss(z1_after, z1_video)
    
    # 2. z2 정렬 손실 - 시간적 압축된 표현도 원본과 일치하도록
    loss_z2 = F.mse_loss(z2_after, z2_video)
    
    # 3. 비디오 재구성 손실 - 슬롯 기반 모듈이 생성한 비디오가 원본과 유사하도록
    loss_video = F.mse_loss(video_from_afterimage, target_video)
    
    # 4. 시간적 일관성 손실 - 연속된 프레임 간의 급격한 변화 방지
    loss_temporal = 0.0
    if delta > 0:
        num_frames = video_from_afterimage.shape[2]
        for t in range(1, num_frames):
            frame_t = video_from_afterimage[:, :, t]
            frame_prev = video_from_afterimage[:, :, t-1]
            loss_temporal += F.l1_loss(frame_t, frame_prev)
        loss_temporal /= (num_frames - 1)
    
    # 총 손실
    total_loss = alpha * loss_z1 + beta * loss_z2 + gamma * loss_video + delta * loss_temporal
    
    # 손실 사전
    loss_dict = {
        'z1': loss_z1.item(),
        'z2': loss_z2.item(),
        'video': loss_video.item(),
        'temporal': loss_temporal if isinstance(loss_temporal, float) else loss_temporal.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict

def compute_loss_integrated(outputs, original_video, alpha=1.0, beta=1.0, gamma=0.2, delta=0.05, lambda_vae=0.5):
    """
    AfterimageVAE_IntegratedTraining 모델의 손실 계산
    
    Args:
        outputs (dict): 모델의 출력 (딕셔너리)
        original_video (torch.Tensor): 원본 비디오 [B, C, T, H, W]
        alpha (float): z1 손실 가중치
        beta (float): z2 손실 가중치
        gamma (float): 슬롯 비디오 재구성 손실 가중치
        delta (float): 시간적 일관성 손실 가중치
        lambda_vae (float): VideoVAE 재구성 손실 가중치
        
    Returns:
        tuple:
            - total_loss (torch.Tensor): 총 손실
            - loss_dict (dict): 개별 손실 항목 사전
    """
    # 출력 추출
    slot_video = outputs['slot_video']
    after_recon = outputs['after_recon']
    z1_after = outputs['z1_after']
    z2_after = outputs['z2_after']
    z1_video = outputs['z1_video']
    z2_video = outputs['z2_video']
    video_recon = outputs['video_recon']
    
    # 1. z1 정렬 손실 - 확장된 비디오의 z1이 원본 비디오의 z1과 일치하도록
    loss_z1 = F.mse_loss(z1_after, z1_video)
    
    # 2. z2 정렬 손실 - 시간적 압축된 표현도 원본과 일치하도록
    loss_z2 = F.mse_loss(z2_after, z2_video)
    
    # 3. 슬롯 비디오 재구성 손실 - 슬롯 기반 모듈이 생성한 비디오가 원본과 유사하도록
    loss_slot_video = F.mse_loss(slot_video, original_video)
    
    # 4. VideoVAE 재구성 손실 - VideoVAE가 원본 비디오를 잘 재구성하도록
    loss_video_recon = F.mse_loss(video_recon, original_video)
    
    # 5. 최종 출력 재구성 손실 - z2_after를 디코딩한 결과가 원본과 유사하도록
    loss_after_recon = F.mse_loss(after_recon, original_video)
    
    # 6. 시간적 일관성 손실 - 연속된 프레임 간의 급격한 변화 방지
    loss_temporal = 0.0
    if delta > 0:
        num_frames = slot_video.shape[2]
        for t in range(1, num_frames):
            frame_t = slot_video[:, :, t]
            frame_prev = slot_video[:, :, t-1]
            loss_temporal += F.l1_loss(frame_t, frame_prev)
        loss_temporal /= (num_frames - 1)
    
    # 총 손실
    total_loss = (
        alpha * loss_z1 + 
        beta * loss_z2 + 
        gamma * loss_slot_video + 
        delta * loss_temporal +
        loss_after_recon +
        lambda_vae * loss_video_recon
    )
    
    # 손실 사전
    loss_dict = {
        'z1': loss_z1.item(),
        'z2': loss_z2.item(),
        'slot_video': loss_slot_video.item(),
        'after_recon': loss_after_recon.item(),
        'video_recon': loss_video_recon.item(),
        'temporal': loss_temporal if isinstance(loss_temporal, float) else loss_temporal.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict

def save_config(config, checkpoint_dir):
    """
    모델 학습 설정을 저장합니다.
    
    Args:
        config: 학습 설정 객체
        checkpoint_dir: 체크포인트 디렉토리
    """
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=4)
        
def save_results(results, checkpoint_dir, filename="results.json"):
    """
    실험 결과를 저장합니다.
    
    Args:
        results: 결과 딕셔너리
        checkpoint_dir: 체크포인트 디렉토리
        filename: 결과 파일 이름
    """
    results_path = os.path.join(checkpoint_dir, filename)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)