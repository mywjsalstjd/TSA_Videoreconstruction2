# -*- coding: utf-8 -*-
import os
import argparse
import json
import torch
import torch.optim as optim
from tqdm import tqdm
import datetime

from models import AfterimageVAE_PretrainedVAE
from utils import (
    get_data_loaders,
    check_and_create_dirs,
    setup_logging,
    create_checkpoint_dir,
    create_results_dir,
    compute_loss_pretrained_vae,
    evaluate_pretrained_vae_model,
    save_comparison_gif_pretrained,
    plot_training_curves_pretrained,
    save_config,
    save_results
)
from configs import get_pretrained_vae_config

def parse_args():
    """
    명령행 인수를 파싱합니다.
    
    Returns:
        argparse.Namespace: 파싱된 인수
    """
    # 기본 설정 가져오기
    default_config = get_pretrained_vae_config()
    
    # 실제 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='AfterimageVAE_PretrainedVAE 학습')
    
    # 데이터 관련 설정
    parser.add_argument('--data_dir', type=str, default=default_config.data_dir, help='데이터셋 디렉토리')
    parser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='배치 크기')
    parser.add_argument('--num_workers', type=int, default=default_config.num_workers, help='데이터 로딩 워커 수')
    
    # 모델 관련 설정
    parser.add_argument('--pretrained_vae_path', type=str, required=True, help='사전 학습된 VideoVAE 경로')
    parser.add_argument('--in_channels', type=int, default=default_config.in_channels, help='입력 채널 수')
    parser.add_argument('--latent_dim', type=int, default=default_config.latent_dim, help='잠재 표현 차원')
    parser.add_argument('--base_channels', type=int, default=default_config.base_channels, help='기본 채널 수')
    parser.add_argument('--num_frames', type=int, default=default_config.num_frames, help='프레임 수')
    parser.add_argument('--resolution', type=int, nargs=2, default=default_config.resolution, help='해상도 (H, W)')
    
    # 학습 관련 설정
    parser.add_argument('--lr', type=float, default=default_config.lr, help='학습률')
    parser.add_argument('--epochs', type=int, default=default_config.epochs, help='에폭 수')
    parser.add_argument('--alpha', type=float, default=default_config.alpha, help='z1 손실 가중치')
    parser.add_argument('--beta', type=float, default=default_config.beta, help='z2 손실 가중치')
    parser.add_argument('--gamma', type=float, default=default_config.gamma, help='비디오 재구성 손실 가중치')
    parser.add_argument('--delta', type=float, default=default_config.delta, help='시간적 일관성 손실 가중치')
    parser.add_argument('--val_freq', type=int, default=default_config.val_freq, help='검증 빈도 (에폭)')
    parser.add_argument('--save_freq', type=int, default=default_config.save_freq, help='저장 빈도 (에폭)')
    parser.add_argument('--save_all_epochs', default=default_config.save_all_epochs, action='store_false', help='모든 에폭의 체크포인트 저장 여부 (기본: False)')
    
    # 시스템 관련 설정
    parser.add_argument('--device', type=str, default=default_config.device, help='학습 장치 (cuda 또는 cpu)')
    parser.add_argument('--gpu_id', type=int, default=default_config.gpu_id, help='사용할 GPU ID')
    parser.add_argument('--checkpoint_dir', type=str, default=default_config.checkpoint_dir, help='체크포인트 기본 디렉토리')
    parser.add_argument('--results_dir', type=str, default=default_config.results_dir, help='결과 기본 디렉토리')
    
    return parser.parse_args()

def train_pretrained_vae():
    """
    AfterimageVAE_PretrainedVAE 모델 학습 메인 함수
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
    
    # 사전 학습된 VideoVAE 경로 확인
    if not os.path.exists(args.pretrained_vae_path):
        raise FileNotFoundError(f"사전 학습된 VideoVAE 모델을 찾을 수 없습니다: {args.pretrained_vae_path}")
    
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
    model_name = "AfterimageVAE_PretrainedVAE"
    
    # 체크포인트 디렉토리 생성
    checkpoint_dir = create_checkpoint_dir(args.checkpoint_dir, model_name)
    
    # 같은 타임스탬프로 결과 디렉토리 생성
    timestamp = os.path.basename(checkpoint_dir)
    results_dir = create_results_dir(args.results_dir, model_name, timestamp)
    
    # 로거 설정
    logger = setup_logging(checkpoint_dir)
    logger.info("Starting AfterimageVAE_PretrainedVAE training...")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    logger.info(f"Results will be saved to: {results_dir}")
    
    # 설정 저장 (체크포인트와 결과 디렉토리 모두에)
    save_config(args, checkpoint_dir)
    save_config(args, results_dir)
    
    # 모델 생성
    model = AfterimageVAE_PretrainedVAE(
        pretrained_vae_path=args.pretrained_vae_path,
        in_channels=args.in_channels,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        num_frames=args.num_frames,
        resolution=tuple(args.resolution)
    ).to(device)
    
    # 학습 가능한 파라미터 수 계산
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 최적화기 설정
    optimizer = optim.Adam([
        {'params': model.slot_expander.parameters()},
        {'params': model.video_encoder.parameters()},
        {'params': model.temporal_encoder.parameters()}
    ], lr=args.lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 학습 기록
    history = {
        'train_loss': [],
        'train_loss_components': [],
        'val_metrics': [],
        'best_val_metrics': None,
        'best_epoch': 0
    }
    
    best_val_mse = float('inf')
    
    # 학습 시작
    for epoch in range(args.epochs):
        # 훈련 단계
        model.train()
        total_loss = 0.0
        epoch_loss_components = {'z1': 0.0, 'z2': 0.0, 'video': 0.0, 'temporal': 0.0}
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in train_pbar:
            afterimage = batch['afterimage'].to(device)
            video = batch['video'].to(device)
            
            optimizer.zero_grad()
            
            # 손실 계산
            loss, loss_dict = compute_loss_pretrained_vae(
                model, afterimage, video, 
                alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta
            )
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            # 손실 누적
            total_loss += loss_dict['total']
            for k, v in loss_dict.items():
                if k != 'total':
                    epoch_loss_components[k] += v
            
            train_pbar.set_postfix({'loss': loss_dict['total']})
        
        # 평균 훈련 손실
        avg_loss = total_loss / len(train_loader)
        avg_components = {k: v / len(train_loader) for k, v in epoch_loss_components.items()}
        
        history['train_loss'].append(avg_loss)
        history['train_loss_components'].append(avg_components)
        
        # 손실 로깅
        log_str = f"Epoch {epoch+1}/{args.epochs}, Loss={avg_loss:.4f}, "
        log_str += ", ".join([f"{k}={v:.4f}" for k, v in avg_components.items()])
        logger.info(log_str)
        
        # 검증 단계 (val_freq에 따라)
        if (epoch + 1) % args.val_freq == 0:
            val_metrics = evaluate_pretrained_vae_model(model, val_loader, device=device)
            history['val_metrics'].append(val_metrics)
            
            # 검증 지표 로깅
            logger.info(f"  Val: MSE={val_metrics['MSE']:.6f}, PSNR={val_metrics['PSNR']:.2f}, "
                      f"SSIM={val_metrics['SSIM']:.4f}")
            
            # 학습률 조정
            scheduler.step(val_metrics['MSE'])
            
            # 최고 모델 저장
            if val_metrics['MSE'] < best_val_mse:
                best_val_mse = val_metrics['MSE']
                history['best_val_metrics'] = val_metrics
                history['best_epoch'] = epoch
                
                # 체크포인트 저장
                model.save(os.path.join(checkpoint_dir, "best_model.pth"))
                logger.info(f"Saved best model at epoch {epoch+1}")
                
                # 시각화 생성 (결과 디렉토리에 저장)
                if (epoch + 1) % args.save_freq == 0 or epoch == 0:
                    out_gif_path = os.path.join(results_dir, f"epoch_{epoch+1}_vis.gif")
                    save_comparison_gif_pretrained(model, val_loader, device=device, out_path=out_gif_path)
        
        # 주기적 모델 저장 부분 수정 (save_freq에 따라)
        if (epoch + 1) % args.save_freq == 0:
            # 시각화만 수행하고 체크포인트는 저장하지 않음
            out_gif_path = os.path.join(results_dir, f"epoch_{epoch+1}_vis.gif")
            
            # 해당 모델에 맞는 시각화 함수 호출
            if "VideoVAE" in model_name:
                # VideoVAE 모델은 시각화 별도 구현 필요
                pass
            elif "PretrainedVAE" in model_name:
                save_comparison_gif_pretrained(model, val_loader, device=device, out_path=out_gif_path)
            else:  # IntegratedTraining
                save_comparison_gif_integrated(model, val_loader, device=device, out_path=out_gif_path)
            
            logger.info(f"Saved visualization at epoch {epoch+1}")
            
            # 에폭별 체크포인트는 save_all_epochs 옵션이 True인 경우에만 저장
            if args.save_all_epochs:
                model.save(os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))
                logger.info(f"Saved model checkpoint at epoch {epoch+1}")
    
    # 학습 곡선 그래프 생성 (결과 디렉토리에 저장)
    plot_training_curves_pretrained(history, results_dir)
    logger.info("Training completed. Generating final visualizations and evaluation...")
    
    # 최종 모델 저장
    model.save(os.path.join(checkpoint_dir, "final_model.pth"))
    logger.info(f"Saved final model after {args.epochs} epochs")
    
    # 최고 모델 로드
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    model.load(best_model_path)
    
    # 테스트 평가
    test_metrics = evaluate_pretrained_vae_model(model, test_loader, device=device, desc="Test")
    
    # 테스트 지표 출력
    logger.info("=== Test Metrics ===")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.6f}")
    
    # 최종 시각화 생성 (결과 디렉토리에 저장)
    out_gif_path = os.path.join(results_dir, "final_comparison.gif")
    save_comparison_gif_pretrained(model, test_loader, device=device, out_path=out_gif_path, num_samples=5)
    logger.info(f"GIF saved => {out_gif_path}")
    
    # 결과 저장
    results = {
        "test_metrics": test_metrics,
        "best_val_metrics": history["best_val_metrics"],
        "best_epoch": history["best_epoch"],
        "pretrained_vae_path": args.pretrained_vae_path,
        "hyperparameters": {
            "latent_dim": args.latent_dim,
            "base_channels": args.base_channels,
            "lr": args.lr,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "delta": args.delta,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "resolution": list(args.resolution)
        }
    }
    
    save_results(results, results_dir)
    logger.info(f"Results saved to {results_dir}")
    
    return model, best_model_path, results_dir

if __name__ == "__main__":
    train_pretrained_vae()