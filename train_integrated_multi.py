# -*- coding: utf-8 -*-
import os
import argparse
import json
import torch
import torch.optim as optim
from tqdm import tqdm
import datetime
import sys
import subprocess
from multiprocessing import Process

from models import AfterimageVAE_IntegratedTraining
from utils import (
    get_data_loaders,
    get_data_loaders_no_shuffle,
    check_and_create_dirs,
    setup_logging,
    create_checkpoint_dir,
    create_results_dir,
    compute_loss_integrated,
    evaluate_integrated_model,
    save_comparison_gif_integrated,
    plot_training_curves_integrated,
    save_config,
    save_results
)
from configs import get_integrated_config

# 데이터세트 구성 정보
AFTERIMAGE_DATASETS = [
    # "mnist_afterimages.npy",         # 기본 잔상 이미지
    "mnist_interval1_afterimages.npy", # interval1 잔상 이미지
    "mnist_interval2_afterimages.npy", # interval2 잔상 이미지 
    "mnist_interval5_afterimages.npy", # interval5 잔상 이미지
    "mnist_interval10_afterimages.npy" # interval10 잔상 이미지
]

def parse_args():
    """
    명령행 인수를 파싱합니다.
    
    Returns:
        argparse.Namespace: 파싱된 인수
    """
    # 기본 설정 가져오기
    default_config = get_integrated_config()
    
    # 실제 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='AfterimageVAE_IntegratedTraining 학습')
    
    # 데이터 관련 설정
    parser.add_argument('--data_dir', type=str, default=default_config.data_dir, help='데이터셋 디렉토리')
    parser.add_argument('--afterimage_dataset', type=str, help='사용할 잔상 이미지 데이터세트 (기본은 GPU ID에 따라 결정)')
    parser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='배치 크기')
    parser.add_argument('--num_workers', type=int, default=default_config.num_workers, help='데이터 로딩 워커 수')
    
    # 모델 관련 설정
    parser.add_argument('--pretrained_vae_path', type=str, default=default_config.pretrained_vae_path, help='사전 학습된 VideoVAE 경로 (선택적)')
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
    parser.add_argument('--gamma', type=float, default=default_config.gamma, help='슬롯 비디오 재구성 손실 가중치')
    parser.add_argument('--delta', type=float, default=default_config.delta, help='시간적 일관성 손실 가중치')
    parser.add_argument('--lambda_vae', type=float, default=default_config.lambda_vae, help='VideoVAE 재구성 손실 가중치')
    parser.add_argument('--val_freq', type=int, default=default_config.val_freq, help='검증 빈도 (에폭)')
    parser.add_argument('--save_freq', type=int, default=default_config.save_freq, help='저장 빈도 (에폭)')
    parser.add_argument('--save_all_epochs', default=default_config.save_all_epochs, action='store_false', help='모든 에폭의 체크포인트 저장 여부 (기본: False)')
    
    # 시스템 관련 설정
    parser.add_argument('--device', type=str, default=default_config.device, help='학습 장치 (cuda 또는 cpu)')
    parser.add_argument('--gpu_id', type=int, default=default_config.gpu_id, help='사용할 GPU ID')
    parser.add_argument('--parallel', action='store_true', help='다중 GPU 병렬 학습 모드')
    parser.add_argument('--checkpoint_dir', type=str, default=default_config.checkpoint_dir, help='체크포인트 기본 디렉토리')
    parser.add_argument('--results_dir', type=str, default=default_config.results_dir, help='결과 기본 디렉토리')
    
    args = parser.parse_args()
    
    # GPU ID에 따라 데이터세트 자동 설정 (--afterimage_dataset이 지정되지 않은 경우)
    if args.afterimage_dataset is None and args.parallel is False:
        gpu_id = args.gpu_id
        if gpu_id < len(AFTERIMAGE_DATASETS):
            args.afterimage_dataset = AFTERIMAGE_DATASETS[gpu_id]
        else:
            args.afterimage_dataset = AFTERIMAGE_DATASETS[0]
    
    return args

def get_custom_data_loaders(args, afterimage_filename):
    """
    커스텀 잔상 이미지 데이터세트에 대한 데이터 로더를 생성합니다.
    
    Args:
        args: 학습 설정 객체
        afterimage_filename: 잔상 이미지 파일명
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 기본 데이터셋 디렉토리 확인
    base_dir = args.data_dir
    os.makedirs(os.path.join(base_dir, "MovingMNIST"), exist_ok=True)
    
    # 잔상 이미지 및 비디오 경로 설정
    afterimage_path = os.path.join(base_dir, "MovingMNIST", afterimage_filename)
    video_path = os.path.join(base_dir, "MovingMNIST", "mnist_test_seq.npy")
    
    # 파일 존재 확인
    if not os.path.exists(afterimage_path):
        alternative_afterimage_path = afterimage_filename
        if os.path.exists(alternative_afterimage_path):
            afterimage_path = alternative_afterimage_path
        else:
            raise FileNotFoundError(f"잔상 이미지 파일을 찾을 수 없습니다: {afterimage_path}")
    
    if not os.path.exists(video_path):
        alternative_video_path = "mnist_test_seq.npy"
        if os.path.exists(alternative_video_path):
            video_path = alternative_video_path
        else:
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = get_data_loaders_no_shuffle(
        afterimage_path=afterimage_path,
        video_path=video_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    return train_loader, val_loader, test_loader

def train_integrated(args, afterimage_dataset=None):
    """
    AfterimageVAE_IntegratedTraining 모델 학습 메인 함수
    
    Args:
        args: 학습 설정 객체
        afterimage_dataset: 사용할 잔상 이미지 데이터세트 파일명
    
    Returns:
        tuple: (model, best_model_path, results_dir)
    """
    # 사용할 잔상 이미지 데이터세트 결정
    if afterimage_dataset is None:
        afterimage_dataset = args.afterimage_dataset
    
    # 장치 설정
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        torch.cuda.set_device(device)
        print(f"GPU {args.gpu_id}에서 {afterimage_dataset} 데이터세트 학습 시작 - {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print(f"CPU에서 {afterimage_dataset} 데이터세트 학습 시작")
    
    # 데이터 로더 생성 (커스텀 데이터세트 사용)
    train_loader, val_loader, test_loader = get_custom_data_loaders(args, afterimage_dataset)
    
    # 모델 이름 설정 (데이터세트 정보 포함)
    dataset_name = afterimage_dataset.replace('.npy', '').replace('mnist_', '')
    model_name = f"AfterimageVAE_IntegratedTraining_{dataset_name}"
    
    # 체크포인트 디렉토리 생성
    checkpoint_dir = create_checkpoint_dir(args.checkpoint_dir, model_name)
    
    # 같은 타임스탬프로 결과 디렉토리 생성
    timestamp = os.path.basename(checkpoint_dir)
    results_dir = create_results_dir(args.results_dir, model_name, timestamp)
    
    # 로거 설정
    logger = setup_logging(checkpoint_dir)
    logger.info(f"Starting {model_name} training with {afterimage_dataset}...")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    logger.info(f"Results will be saved to: {results_dir}")
    
    # 설정 저장 (체크포인트와 결과 디렉토리 모두에)
    # 사용한 데이터세트 정보도 추가
    args_dict = vars(args).copy()
    args_dict['afterimage_dataset'] = afterimage_dataset
    
    with open(os.path.join(checkpoint_dir, "config.json"), 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    with open(os.path.join(results_dir, "config.json"), 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    # 모델 생성
    model = AfterimageVAE_IntegratedTraining(
        in_channels=args.in_channels,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        num_frames=args.num_frames,
        resolution=tuple(args.resolution),
        pretrained_vae_path=args.pretrained_vae_path,
        device=device
    )
    
    # 사전 학습된 VideoVAE를 사용하는 경우 로그
    if args.pretrained_vae_path:
        if os.path.exists(args.pretrained_vae_path):
            logger.info(f"Using pretrained VideoVAE weights from: {args.pretrained_vae_path}")
        else:
            logger.warning(f"Specified pretrained VAE path does not exist: {args.pretrained_vae_path}")
            logger.warning("Training will continue with randomly initialized weights")
    
    model = model.to(device)
    
    # 학습 가능한 파라미터 수 계산
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 최적화기 설정 - 모든 파라미터 최적화 (함께 학습)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 스케줄러 설정 변경: SSIM은 높을수록 좋으므로 mode='max'로 설정
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # 학습 기록
    history = {
        'train_loss': [],
        'train_loss_components': [],
        'val_metrics': [],
        'best_val_metrics': None,
        'best_epoch': 0
    }
    
    # SSIM 기준으로 최적 모델 저장 (높을수록 좋음)
    best_val_ssim = 0.0
    
    # 학습 시작
    for epoch in range(args.epochs):
        # 훈련 단계
        model.train()
        total_loss = 0.0
        epoch_loss_components = {'z1': 0.0, 'z2': 0.0, 'slot_video': 0.0, 'after_recon': 0.0, 'video_recon': 0.0, 'temporal': 0.0}
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in train_pbar:
            afterimage = batch['afterimage'].to(device)
            video = batch['video'].to(device)
            
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(afterimage, video)
            
            # 손실 계산
            loss, loss_dict = compute_loss_integrated(
                outputs, video, 
                alpha=args.alpha, beta=args.beta, gamma=args.gamma, 
                delta=args.delta, lambda_vae=args.lambda_vae
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
            val_metrics = evaluate_integrated_model(model, val_loader, device=device)
            history['val_metrics'].append(val_metrics)
            
            # 검증 지표 로깅
            logger.info(f"  Val (Final): MSE={val_metrics['final']['MSE']:.6f}, PSNR={val_metrics['final']['PSNR']:.2f}, "
                       f"SSIM={val_metrics['final']['SSIM']:.4f}")
            logger.info(f"  Val (Slot): MSE={val_metrics['slot']['MSE']:.6f}, PSNR={val_metrics['slot']['PSNR']:.2f}, "
                       f"SSIM={val_metrics['slot']['SSIM']:.4f}")
            logger.info(f"  Val (VAE): MSE={val_metrics['vae']['MSE']:.6f}, PSNR={val_metrics['vae']['PSNR']:.2f}, "
                       f"SSIM={val_metrics['vae']['SSIM']:.4f}")
            
            # 현재 SSIM 값 (final 출력 기준)
            current_ssim = val_metrics['final']['SSIM']
            
            # 학습률 조정 (SSIM 기준)
            scheduler.step(current_ssim)
            
            # 최고 모델 저장 (SSIM 기준으로 변경)
            if current_ssim > best_val_ssim:
                best_val_ssim = current_ssim
                history['best_val_metrics'] = val_metrics
                history['best_epoch'] = epoch
                
                # 체크포인트 저장
                model.save(os.path.join(checkpoint_dir, "best_model.pth"))
                logger.info(f"Saved best model at epoch {epoch+1} with SSIM={current_ssim:.4f}")
                
                # 시각화 생성 (결과 디렉토리에 저장)
                if (epoch + 1) % args.save_freq == 0 or epoch == 0:
                    out_gif_path = os.path.join(results_dir, f"epoch_{epoch+1}_vis.gif")
                    save_comparison_gif_integrated(model, val_loader, device=device, out_path=out_gif_path)
        
        # 주기적 모델 저장 부분 수정 (save_freq에 따라)
        if (epoch + 1) % args.save_freq == 0:
            # 시각화 생성
            out_gif_path = os.path.join(results_dir, f"epoch_{epoch+1}_vis.gif")
            save_comparison_gif_integrated(model, val_loader, device=device, out_path=out_gif_path)
            logger.info(f"Saved visualization at epoch {epoch+1}")
            
            # 에폭별 체크포인트는 save_all_epochs 옵션이 True인 경우에만 저장
            if args.save_all_epochs:
                model.save(os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))
                logger.info(f"Saved model checkpoint at epoch {epoch+1}")
    
    # 학습 곡선 그래프 생성 (결과 디렉토리에 저장)
    plot_training_curves_integrated(history, results_dir)
    logger.info("Training completed. Generating final visualizations and evaluation...")
    
    # 최종 모델 저장
    model.save(os.path.join(checkpoint_dir, "final_model.pth"))
    logger.info(f"Saved final model after {args.epochs} epochs")
    
    # 최고 모델 로드
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    model.load(best_model_path)
    
    # 테스트 평가
    test_metrics = evaluate_integrated_model(model, test_loader, device=device, desc="Test")
    
    # 테스트 지표 출력
    logger.info("\n=== Test Metrics ===")
    logger.info("Final Output (Slot → VideoVAE):")
    for k, v in test_metrics['final'].items():
        logger.info(f"  {k}: {v:.6f}")
    
    logger.info("\nSlot Video:")
    for k, v in test_metrics['slot'].items():
        logger.info(f"  {k}: {v:.6f}")
    
    logger.info("\nVideoVAE:")
    for k, v in test_metrics['vae'].items():
        logger.info(f"  {k}: {v:.6f}")
    
    # 최종 시각화 생성 (결과 디렉토리에 저장)
    out_gif_path = os.path.join(results_dir, "final_comparison.gif")
    save_comparison_gif_integrated(model, test_loader, device=device, out_path=out_gif_path, num_samples=5)
    logger.info(f"GIF saved => {out_gif_path}")
    
    # 결과 저장
    results = {
        "dataset": afterimage_dataset,
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
            "lambda_vae": args.lambda_vae,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "resolution": list(args.resolution)
        }
    }
    
    save_results(results, results_dir)
    logger.info(f"Results saved to {results_dir}")
    
    return model, best_model_path, results_dir

def run_training_on_gpu(gpu_id, dataset):
    """각 GPU에서 독립적으로 학습을 실행하는 함수"""
    args = parse_args()
    args.gpu_id = gpu_id
    args.afterimage_dataset = dataset
    train_integrated(args)

def main():
    # 가장 먼저 멀티프로세싱 시작 방법 설정
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    args = parse_args()
    
    # 병렬 모드인 경우, 모든 GPU에 대해 서로 다른 데이터세트로 학습 실행
    if args.parallel:
        processes = []
        num_gpus = min(torch.cuda.device_count(), len(AFTERIMAGE_DATASETS))
        
        print(f"병렬 모드 활성화 - {num_gpus}개 GPU에서 학습 시작")
        
        for gpu_id in range(num_gpus):
            dataset = AFTERIMAGE_DATASETS[gpu_id]
            p = multiprocessing.Process(target=run_training_on_gpu, args=(gpu_id, dataset))
            p.start()
            processes.append(p)
        
        # 모든 프로세스가 완료될 때까지 대기
        for p in processes:
            p.join()
        
        print("모든 GPU에서 학습 완료!")
    
    # 단일 GPU 모드
    else:
        train_integrated(args)

if __name__ == "__main__":
    # 멀티프로세싱 시작 방법 설정
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()