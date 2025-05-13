# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from models.modules import (
    TemporalAwareSpatialEncoder,
    TemporalEncoder,
    TemporalDecoder,
    TemporalAwareSpatialDecoder
)

class VideoVAE(nn.Module):
    """
    비디오 VAE 모델 - 비디오 인코딩 및 디코딩을 위한 기본 모델
    
    두 단계의 인코딩/디코딩 프로세스:
    1. 시공간 인코딩/디코딩 (z1 표현)
    2. 시간 압축/확장 (z2 표현)
    
    Args:
        in_channels (int): 입력 채널 수
        latent_dim (int): 잠재 표현 차원
        base_channels (int): 기본 채널 수
    """
    def __init__(self, in_channels=1, latent_dim=4, base_channels=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 비디오 인코더-디코더
        self.video_encoder = TemporalAwareSpatialEncoder(in_channels, latent_dim, base_channels)
        self.temporal_encoder = TemporalEncoder(latent_dim, latent_dim)
        self.temporal_decoder = TemporalDecoder(latent_dim, latent_dim)
        self.video_decoder = TemporalAwareSpatialDecoder(latent_dim, in_channels, base_channels)
    
    def encode(self, x):
        """
        비디오 인코딩 함수
        
        Args:
            x (torch.Tensor): 입력 비디오 [B, C, T, H, W]
            
        Returns:
            tuple:
                - z1 (torch.Tensor): 시공간 잠재 표현 [B, latent_dim, T, H/8, W/8]
                - z2 (torch.Tensor): 시간 압축된 잠재 표현 [B, latent_dim, T/4, H/8, W/8]
        """
        # 비디오 인코딩
        z1 = self.video_encoder(x)
        z2 = self.temporal_encoder(z1)
        return z1, z2
    
    def decode(self, z2):
        """
        비디오 디코딩 함수
        
        Args:
            z2 (torch.Tensor): 시간 압축된 잠재 표현 [B, latent_dim, T/4, H/8, W/8]
            
        Returns:
            torch.Tensor: 재구성된 비디오 [B, C, T, H, W]
        """
        # 시간 디코딩
        z1_recon = self.temporal_decoder(z2)
        # 공간 디코딩
        x_recon = self.video_decoder(z1_recon)
        return x_recon
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 비디오 [B, C, T, H, W]
            
        Returns:
            tuple:
                - x_recon (torch.Tensor): 재구성된 비디오 [B, C, T, H, W]
                - z1 (torch.Tensor): 시공간 잠재 표현
                - z2 (torch.Tensor): 시간 압축된 잠재 표현
        """
        # 인코딩
        z1, z2 = self.encode(x)
        # 디코딩
        x_recon = self.decode(z2)
        return x_recon, z1, z2
        
    def save(self, path):
        """모델 저장"""
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        """모델 로드"""
        self.load_state_dict(torch.load(path, map_location='cpu'))