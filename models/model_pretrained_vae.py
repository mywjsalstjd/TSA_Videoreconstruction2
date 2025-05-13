# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from models.video_vae import VideoVAE
from models.modules import (
    TemporalAwareSpatialEncoder,
    TemporalEncoder,
    SlotBasedAfterimageToVideo
)

class AfterimageVAE_PretrainedVAE(nn.Module):
    """
    사전 학습된 VideoVAE를 사용하는 잔상 기반 비디오 생성 모델
    
    이 모델에서는 VideoVAE 가중치가 고정되며, 슬롯 어텐션 기반 모듈만 학습됩니다.
    2단계 접근법으로 학습 부담을 줄이고 안정성을 향상시킵니다.
    
    학습 과정:
    1. 잔상 이미지 → 슬롯 비디오 → 잠재 표현(z1, z2)
    2. 고정된 VAE의 잠재 표현과 정렬하도록 학습
    
    Args:
        pretrained_vae_path (str): 사전 학습된 VideoVAE 모델 경로
        in_channels (int): 입력 채널 수
        latent_dim (int): 잠재 표현 차원
        base_channels (int): 기본 채널 수
        num_frames (int): 생성할 프레임 수
        resolution (tuple): 입력 해상도 (H, W)
    """
    def __init__(self, pretrained_vae_path, in_channels=1, latent_dim=4, base_channels=32, num_frames=20, resolution=(64, 64)):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.resolution = resolution
        
        # 1. 사전 학습된 VideoVAE 로드 (가중치 고정)
        self.video_vae = VideoVAE(in_channels, latent_dim, base_channels)
        self.video_vae.load(pretrained_vae_path)
        self.video_vae.eval()
        for param in self.video_vae.parameters():
            param.requires_grad = False
        
        # 2. 잔상 이미지를 비디오로 확장하는 슬롯 기반 모듈
        self.slot_expander = SlotBasedAfterimageToVideo(
            in_channels=in_channels, 
            out_channels=in_channels, 
            num_frames=num_frames, 
            hidden_dim=base_channels*2,
            resolution=resolution
        )
        
        # 3. 비디오 인코더 (z1 생성) - VideoVAE와 동일한 구조지만 학습 가능
        self.video_encoder = TemporalAwareSpatialEncoder(in_channels, latent_dim, base_channels)
        
        # 4. 시간 인코더 (z2 생성) - VideoVAE와 동일한 구조지만 학습 가능
        self.temporal_encoder = TemporalEncoder(latent_dim, latent_dim)
        
    def forward(self, x):
        """
        잔상 이미지를 비디오로 변환
        
        Args:
            x (torch.Tensor): 잔상 이미지 [B, C, H, W]
            
        Returns:
            tuple:
                - x_recon (torch.Tensor): 재구성된 비디오 [B, C, T, H, W]
                - z1_after (torch.Tensor): 확장된 비디오의 z1 표현
                - z2_after (torch.Tensor): 확장된 비디오의 z2 표현
                - video_from_afterimage (torch.Tensor): 슬롯 기반으로 확장된 비디오
                - attention_masks (torch.Tensor): 각 프레임의 어텐션 마스크
        """
        # 1. 잔상 이미지를 비디오로 확장
        video_from_afterimage, attention_masks = self.slot_expander(x)
        
        # 2. 확장된 비디오를 인코딩하여 z1 생성
        z1_after = self.video_encoder(video_from_afterimage)
        
        # 3. z1에서 z2 생성
        z2_after = self.temporal_encoder(z1_after)
        
        # 4. 고정된 VideoVAE 디코더로 비디오 재구성
        with torch.no_grad():
            x_recon = self.video_vae.decode(z2_after)
        
        return x_recon, z1_after, z2_after, video_from_afterimage, attention_masks
    
    def get_reference_latents(self, video):
        """
        원본 비디오에서 참조용 잠재 표현 추출 (훈련용)
        
        Args:
            video (torch.Tensor): 원본 비디오 [B, C, T, H, W]
            
        Returns:
            tuple:
                - z1_video (torch.Tensor): 원본 비디오의 z1 표현
                - z2_video (torch.Tensor): 원본 비디오의 z2 표현
        """
        with torch.no_grad():
            z1_video, z2_video = self.video_vae.encode(video)
        return z1_video, z2_video
    
    def visualize_slots(self, x):
        """
        슬롯 어텐션 시각화
        
        Args:
            x (torch.Tensor): 잔상 이미지 [B, C, H, W]
            
        Returns:
            torch.Tensor: 어텐션 마스크 [B, 1, T, H, W]
        """
        _, _, _, _, attention_masks = self.forward(x)
        return attention_masks
    
    def save(self, path):
        """모델 저장"""
        torch.save({
            'slot_expander': self.slot_expander.state_dict(),
            'video_encoder': self.video_encoder.state_dict(),
            'temporal_encoder': self.temporal_encoder.state_dict(),
        }, path)
        
    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location='cpu')
        self.slot_expander.load_state_dict(checkpoint['slot_expander'])
        self.video_encoder.load_state_dict(checkpoint['video_encoder'])
        self.temporal_encoder.load_state_dict(checkpoint['temporal_encoder'])