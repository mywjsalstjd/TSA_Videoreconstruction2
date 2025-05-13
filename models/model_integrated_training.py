# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from models.video_vae import VideoVAE
from models.modules import SlotBasedAfterimageToVideo

class AfterimageVAE_IntegratedTraining(nn.Module):
    """
    VideoVAE와 슬롯 어텐션을 함께 학습하는 통합 모델
    
    이 모델은 VideoVAE와 슬롯 어텐션 모듈을 동시에 학습하는 통합적 접근 방식을 사용합니다.
    엔드투엔드 훈련을 통해 모델 구성 요소 간의 더 나은 조정을 가능하게 합니다.
    
    학습 과정:
    1. 슬롯 모듈: 잔상 이미지 → 비디오 프레임 생성
    2. VideoVAE: 슬롯 생성 비디오와 원본 비디오 모두에 대한 인코딩/디코딩
    3. 손실 함수: 재구성 품질, 슬롯 비디오 품질, 잠재 표현 정렬을 모두 고려
    
    Args:
        in_channels (int): 입력 채널 수
        latent_dim (int): 잠재 표현 차원
        base_channels (int): 기본 채널 수
        num_frames (int): 생성할 프레임 수
        resolution (tuple): 입력 해상도 (H, W)
        pretrained_vae_path (str, optional): 사전 학습된 VideoVAE 모델 경로 (선택 사항)
        device (torch.device, optional): 모델이 실행될 장치
    """
    def __init__(self, in_channels=1, latent_dim=4, base_channels=32, num_frames=20, resolution=(64, 64), 
                pretrained_vae_path=None, device=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.resolution = resolution
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. VideoVAE 생성 (함께 학습됨)
        self.video_vae = VideoVAE(in_channels, latent_dim, base_channels)
        
        # 사전 학습된 VideoVAE 가중치 로드 (선택적)
        if pretrained_vae_path is not None:
            self.video_vae.load(pretrained_vae_path)
            print(f"Loaded pretrained VideoVAE from {pretrained_vae_path}")
        
        # 2. 잔상 이미지를 비디오로 확장하는 슬롯 기반 모듈
        self.slot_expander = SlotBasedAfterimageToVideo(
            in_channels=in_channels, 
            out_channels=in_channels, 
            num_frames=num_frames, 
            hidden_dim=base_channels*2,
            resolution=resolution,
            device=self.device
        )
        
    def forward(self, x, original_video=None):
        """
        Args:
            x (torch.Tensor): 잔상 이미지 [B, C, H, W]
            original_video (torch.Tensor, optional): 원본 비디오 [B, C, T, H, W] (학습 시 원본에서 z1, z2 계산용)
            
        Returns:
            dict: 다양한 출력을 포함한 딕셔너리
                - slot_video: 슬롯 모듈이 생성한 비디오
                - attention_masks: 어텐션 마스크
                - z1_after: 슬롯 비디오의 z1 표현
                - z2_after: 슬롯 비디오의 z2 표현
                - after_recon: z2_after를 디코딩한 비디오
                - z1_video (선택): 원본 비디오의 z1 표현
                - z2_video (선택): 원본 비디오의 z2 표현
                - video_recon (선택): 원본 비디오의 재구성
        """
        outputs = {}
        
        # 1. 잔상 이미지를 비디오로 확장
        slot_video, attention_masks = self.slot_expander(x)
        outputs['slot_video'] = slot_video
        outputs['attention_masks'] = attention_masks
        
        # 2. 확장된 비디오를 인코딩하여 z1, z2 생성
        z1_after, z2_after = self.video_vae.encode(slot_video)
        outputs['z1_after'] = z1_after
        outputs['z2_after'] = z2_after
        
        # 3. z2를 디코딩하여 최종 비디오 재구성
        after_recon = self.video_vae.decode(z2_after)
        outputs['after_recon'] = after_recon
        
        # 4. 원본 비디오가 제공된 경우 추가 계산 수행 (학습 시)
        if original_video is not None:
            # 원본 비디오의 인코딩
            z1_video, z2_video = self.video_vae.encode(original_video)
            outputs['z1_video'] = z1_video
            outputs['z2_video'] = z2_video
            
            # 원본 비디오 재구성 (VideoVAE 학습용)
            video_recon = self.video_vae.decode(z2_video)
            outputs['video_recon'] = video_recon
        
        return outputs
    
    def save(self, path):
        """모델 저장"""
        torch.save({
            'video_vae': self.video_vae.state_dict(),
            'slot_expander': self.slot_expander.state_dict(),
        }, path)
        
    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location='cpu')
        self.video_vae.load_state_dict(checkpoint['video_vae'])
        self.slot_expander.load_state_dict(checkpoint['slot_expander'])