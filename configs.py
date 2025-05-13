# -*- coding: utf-8 -*-
import os
import argparse

def get_video_vae_config():
    """
    VideoVAE 모델 학습을 위한 설정을 반환합니다.
    
    Returns:
        argparse.Namespace: 설정 객체
    """
    parser = argparse.ArgumentParser(description='VideoVAE 학습 파라미터')
    
    # 데이터 관련 설정
    parser.add_argument('--data_dir', type=str, default='./datasets', help='데이터셋 디렉토리')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--num_workers', type=int, default=4, help='데이터 로딩 워커 수')
    
    # 모델 관련 설정
    parser.add_argument('--in_channels', type=int, default=1, help='입력 채널 수')
    parser.add_argument('--latent_dim', type=int, default=4, help='잠재 표현 차원')
    parser.add_argument('--base_channels', type=int, default=32, help='기본 채널 수')
    
    # 학습 관련 설정
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--epochs', type=int, default=50, help='에폭 수')
    parser.add_argument('--val_freq', type=int, default=1, help='검증 빈도 (에폭)')
    parser.add_argument('--save_freq', type=int, default=1, help='저장 빈도 (에폭)')
    
    # 시스템 관련 설정
    parser.add_argument('--device', type=str, default='cuda', help='학습 장치 (cuda 또는 cpu)')
    parser.add_argument('--gpu_id', type=int, default=0, help='사용할 GPU ID')
    # 각 config 함수 내에서:
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='체크포인트 기본 디렉토리')
    parser.add_argument('--results_dir', type=str, default='./results', help='결과 기본 디렉토리')
    # 파싱
    args = parser.parse_args([])
    
    return args

def get_pretrained_vae_config():
    """
    AfterimageVAE_PretrainedVAE 모델 학습을 위한 설정을 반환합니다.
    
    Returns:
        argparse.Namespace: 설정 객체
    """
    parser = argparse.ArgumentParser(description='AfterimageVAE_PretrainedVAE 학습 파라미터')
    
    # 데이터 관련 설정
    parser.add_argument('--data_dir', type=str, default='./datasets', help='데이터셋 디렉토리')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--num_workers', type=int, default=4, help='데이터 로딩 워커 수')
    
    # 모델 관련 설정
    parser.add_argument('--pretrained_vae_path', type=str, required=True, help='사전 학습된 VideoVAE 경로')
    parser.add_argument('--in_channels', type=int, default=1, help='입력 채널 수')
    parser.add_argument('--latent_dim', type=int, default=8, help='잠재 표현 차원')
    parser.add_argument('--base_channels', type=int, default=32, help='기본 채널 수')
    parser.add_argument('--num_frames', type=int, default=20, help='프레임 수')
    parser.add_argument('--resolution', type=int, nargs=2, default=[64, 64], help='해상도 (H, W)')
    
    # 학습 관련 설정
    parser.add_argument('--lr', type=float, default=5e-4, help='학습률')
    parser.add_argument('--epochs', type=int, default=50, help='에폭 수')
    parser.add_argument('--alpha', type=float, default=1.0, help='z1 손실 가중치')
    parser.add_argument('--beta', type=float, default=1.0, help='z2 손실 가중치')
    parser.add_argument('--gamma', type=float, default=0.2, help='비디오 재구성 손실 가중치')
    parser.add_argument('--delta', type=float, default=0.05, help='시간적 일관성 손실 가중치')
    parser.add_argument('--val_freq', type=int, default=1, help='검증 빈도 (에폭)')
    parser.add_argument('--save_freq', type=int, default=1, help='저장 빈도 (에폭)')
    
    # 체크포인트 저장 여부를 제어할 매개변수 추가
    parser.add_argument('--save_all_epochs', action='store_false', help='모든 에폭의 체크포인트 저장 여부 (기본: False)')
    
    # 시스템 관련 설정
    parser.add_argument('--device', type=str, default='cuda', help='학습 장치 (cuda 또는 cpu)')
    parser.add_argument('--gpu_id', type=int, default=0, help='사용할 GPU ID')
    # 각 config 함수 내에서:
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='체크포인트 기본 디렉토리')
    parser.add_argument('--results_dir', type=str, default='./results', help='결과 기본 디렉토리')
    # 파싱
    args = parser.parse_args([])
    
    return args

def get_integrated_config():
    """
    AfterimageVAE_IntegratedTraining 모델 학습을 위한 설정을 반환합니다.
    
    Returns:
        argparse.Namespace: 설정 객체
    """
    parser = argparse.ArgumentParser(description='AfterimageVAE_IntegratedTraining 학습 파라미터')
    
    # 데이터 관련 설정
    parser.add_argument('--data_dir', type=str, default='./datasets', help='데이터셋 디렉토리')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--num_workers', type=int, default=4, help='데이터 로딩 워커 수')
    
    # 모델 관련 설정
    parser.add_argument('--pretrained_vae_path', type=str, default=None, help='사전 학습된 VideoVAE 경로 (선택적)')
    parser.add_argument('--in_channels', type=int, default=1, help='입력 채널 수')
    parser.add_argument('--latent_dim', type=int, default=8, help='잠재 표현 차원')
    parser.add_argument('--base_channels', type=int, default=32, help='기본 채널 수')
    parser.add_argument('--num_frames', type=int, default=20, help='프레임 수')
    parser.add_argument('--resolution', type=int, nargs=2, default=[64, 64], help='해상도 (H, W)')
    
    # 학습 관련 설정
    '''
    gamma=0.0일 때: 슬롯 확장기가 원본 비디오와 유사한 비디오를 생성할 필요가 없음
    gamma>0.0일 때: 슬롯 확장기가 원본 비디오와 유사한 비디오를 생성하도록 학습됨

    lambda_vae=1.0일 때: VAE가 원본 비디오를 완벽히 재구성하는 데 집중
    lambda_vae<1.0일 때: VAE가 원본 재구성보다 다른 손실에 더 주의
    '''
    parser.add_argument('--lr', type=float, default=5e-4, help='학습률')
    parser.add_argument('--epochs', type=int, default=100, help='에폭 수')
    parser.add_argument('--alpha', type=float, default=1.0, help='z1 손실 가중치')
    parser.add_argument('--beta', type=float, default=1.0, help='z2 손실 가중치')
    parser.add_argument('--gamma', type=float, default=0.5, help='슬롯 비디오 재구성 손실 가중치') # 0.2 -> 0.5
    parser.add_argument('--delta', type=float, default=0.01, help='시간적 일관성 손실 가중치') # 0.05 -> 0.01
    parser.add_argument('--lambda_vae', type=float, default=0.5, help='VideoVAE 재구성 손실 가중치') # 0.5
    
    parser.add_argument('--val_freq', type=int, default=1, help='검증 빈도 (에폭)')
    parser.add_argument('--save_freq', type=int, default=10, help='저장 빈도 (에폭)')
    
    # 체크포인트 저장 여부를 제어할 매개변수 추가
    parser.add_argument('--save_all_epochs', action='store_false', help='모든 에폭의 체크포인트 저장 여부 (기본: False)')
    
    # 시스템 관련 설정
    parser.add_argument('--device', type=str, default='cuda', help='학습 장치 (cuda 또는 cpu)')
    parser.add_argument('--gpu_id', type=int, default=0, help='사용할 GPU ID')
    
    # 각 config 함수 내에서:
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='체크포인트 기본 디렉토리')
    parser.add_argument('--results_dir', type=str, default='./results', help='결과 기본 디렉토리')
    # 파싱
    args = parser.parse_args([])
    
    return args