# -*- coding: utf-8 -*-
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import datetime

from models import VideoVAE
from utils import (
    get_data_loaders,
    check_and_create_dirs,
    setup_logging,
    create_checkpoint_dir,
    create_results_dir,
    save_config,
    save_results
)
from configs import get_video_vae_config

def parse_args():
    """
    명령행 인수를 파싱합니다.
    
    Returns:
        argparse.Namespace: 파싱된 인수
    """
    # 기본 설정 가져오기
    default_config = get_video_vae_config()
    
    # 실제 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='VideoVAE 학습')
    
    # 데이터 관련 설정
    parser.add_argument('--data_dir', type=str, default=default_config.data_dir, help='데이터셋 디렉토리')
    parser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='배치 크기')
    parser.add_argument('--num_workers', type=int, default=default_config.num_workers, help='데이터 로딩 워커 수')
    
    # 모델 관련 설정
    parser.add_argument('--in_channels', type=int, default=default_config.in_channels, help='입력 채널 수')
    parser.add_argument('--latent_dim', type=int, default=default_config.latent_dim, help='잠재 표현 차원')
    parser.add_argument('--base_channels', type=int, default=default_config.base_channels, help='기본 채널 수')
    
    # 학습 관련 설정
    parser.add_argument('--lr', type=float, default=default_config.lr, help='학습률')
    parser.add_argument('--epochs', type=int, default=default_config.epochs, help='에폭 수')
    parser.add_argument('--val_freq', type=int, default=default_config.val_freq, help='검증 빈도 (에폭)')
    parser.add_argument('--save_freq', type=int, default=default_config.save_freq, help='저장 빈도 (에폭)')
    
    # 시스템 관련 설정
    parser.add_argument('--device', type=str, default=default_config.device, help='학습 장치 (cuda 또는 cpu)')
    parser.add_argument('--gpu_id', type=int, default=default_config.gpu_id, help='사용할 GPU ID')
    parser.add_argument('--checkpoint_dir', type=str, default=default_config.checkpoint_dir, help='체크포인트 기본 디렉토리')
    parser.add_argument('--results_dir', type=str, default=default_config.results_dir, help='결과 기본 디렉토리')
    
    return parser.parse_args()

def train_video_vae():
    """
    VideoVAE 모델 학습 메인 함수
    """
    # 인수 파싱
    args = parse_args()
    
    # 장치 설정
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        torch.cuda.set_device(device)
        print(f"Using GPU: {args.gpu_id} - {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # 데이터셋 경로 확인
    afterimage_path, video_path = check_and_create_dirs(args.data_dir)
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = get_data_loaders(
        afterimage_path=afterimage_path,
        video_path=video_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 모델 이름 설정
    model_name = "VideoVAE"
    
    # 체크포인트 디렉토리 생성
    checkpoint_dir = create_checkpoint_dir(args.checkpoint_dir, model_name)
    
    # 같은 타임스탬프로 결과 디렉토리 생성
    timestamp = os.path.basename(checkpoint_dir)
    results_dir = create_results_dir(args.results_dir, model_name, timestamp)
    
    # 로거 설정
    logger = setup_logging(checkpoint_dir)
    logger.info("Starting VideoVAE training...")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    logger.info(f"Results will be saved to: {results_dir}")
    
    # 설정 저장 (체크포인트와 결과 디렉토리 모두에)
    save_config(args, checkpoint_dir)
    save_config(args, results_dir)
    
    # 모델 생성
    model = VideoVAE(
        in_channels=args.in_channels,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels
    ).to(device)
    
    # 최적화기 및 손실 함수 설정
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 학습 기록
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    # 학습 시작
    for epoch in range(args.epochs):
        # 훈련 단계
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for batch in train_pbar:
            video = batch['video'].to(device)
            
            optimizer.zero_grad()
            
            # 순전파
            video_recon, _, _ = model(video)
            
            # 손실 계산
            loss = nn.MSELoss()(video_recon, video)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            # 손실 누적
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        # 평균 훈련 손실
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # 검증 단계 (val_freq에 따라)
        if (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss = 0.0
            
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
            with torch.no_grad():
                for batch in val_pbar:
                    video = batch['video'].to(device)
                    
                    # 순전파
                    video_recon, _, _ = model(video)
                    
                    # 손실 계산
                    loss = nn.MSELoss()(video_recon, video)
                    
                    # 손실 누적
                    val_loss += loss.item()
                    val_pbar.set_postfix({'loss': loss.item()})
            
            # 평균 검증 손실
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            # 스케줄러 업데이트
            scheduler.step(avg_val_loss)
            
            # 로깅
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # 최고 모델 저장
            if avg_val_loss < history['best_val_loss']:
                history['best_val_loss'] = avg_val_loss
                history['best_epoch'] = epoch
                
                # 최고 모델 저장 (체크포인트 디렉토리)
                model.save(os.path.join(checkpoint_dir, "best_model.pth"))
                logger.info(f"Saved best model at epoch {epoch+1}, Val Loss: {avg_val_loss:.6f}")
        
        # 주기적 모델 저장 (save_freq에 따라)
        if (epoch + 1) % args.save_freq == 0:
            model.save(os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))
            logger.info(f"Saved model checkpoint at epoch {epoch+1}")
    
    # 최종 모델 저장
    model.save(os.path.join(checkpoint_dir, "final_model.pth"))
    logger.info(f"Saved final model after {args.epochs} epochs")
    
    # 테스트 단계
    logger.info("Evaluating model on test set...")
    model.eval()
    test_loss = 0.0
    
    test_pbar = tqdm(test_loader, desc="Test Evaluation", leave=False)
    with torch.no_grad():
        for batch in test_pbar:
            video = batch['video'].to(device)
            
            # 순전파
            video_recon, _, _ = model(video)
            
            # 손실 계산
            loss = nn.MSELoss()(video_recon, video)
            
            # 손실 누적
            test_loss += loss.item()
            test_pbar.set_postfix({'loss': loss.item()})
    
    # 평균 테스트 손실
    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"Test Loss: {avg_test_loss:.6f}")
    
    # 결과 저장 - 결과 디렉토리에 저장
    results = {
        'test_loss': avg_test_loss,
        'best_val_loss': history['best_val_loss'],
        'best_epoch': history['best_epoch'],
        'train_history': history['train_loss'],
        'val_history': history['val_loss'],
        'model_params': {
            'in_channels': args.in_channels,
            'latent_dim': args.latent_dim,
            'base_channels': args.base_channels
        }
    }
    
    save_results(results, results_dir)
    logger.info(f"Results saved to {results_dir}")
    
    # 최고 성능 모델 경로 반환
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    logger.info(f"Best model saved at: {best_model_path}")
    
    return best_model_path, results_dir

if __name__ == "__main__":
    train_video_vae()