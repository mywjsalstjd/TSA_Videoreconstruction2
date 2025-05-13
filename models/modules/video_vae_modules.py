# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAwareSpatialBlock(nn.Module):
    """
    공간 및 시간 특성을 모두 고려하는 컨볼루션 블록
    
    Args:
        in_channels (int): 입력 채널 수
        out_channels (int): 출력 채널 수
        kernel_size (int): 커널 크기
        stride (int): 스트라이드
        padding (int): 패딩
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 공간 컨볼루션 (1,k,k)
        self.conv3d_spatial = nn.Conv3d(in_channels, out_channels, 
                                       kernel_size=(1, kernel_size, kernel_size), 
                                       stride=(1, stride, stride), 
                                       padding=(0, padding, padding))
        
        # 시간 컨볼루션 (3,k,k)
        self.conv3d_temporal = nn.Conv3d(out_channels, out_channels, 
                                        kernel_size=(3, kernel_size, kernel_size), 
                                        stride=(1, 1, 1), 
                                        padding=(1, padding, padding))
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 [B, C, T, H, W]
            
        Returns:
            torch.Tensor: 출력 [B, C_out, T, H', W']
        """
        # 공간 컨볼루션
        h = self.conv3d_spatial(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # 시간 컨볼루션
        h = self.conv3d_temporal(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h

class TemporalAttention(nn.Module):
    """
    시간 차원에 대한 셀프 어텐션 모듈
    
    Args:
        channels (int): 입력 채널 수
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = channels ** -0.5
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 [B, C, T, H, W]
            
        Returns:
            torch.Tensor: 출력 [B, C, T, H, W]
        """
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # 정규화
        h = self.norm(x)
        
        # 공간 평균화 - 시간 차원 유지
        h_flat = h.mean(dim=(-1, -2))  # [B, C, T]
        
        # 어텐션 계산
        q = self.q(h_flat)  # [B, C, T]
        k = self.k(h_flat)  # [B, C, T]
        v = self.v(h_flat)  # [B, C, T]
        
        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale  # [B, T, T]
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, C, T]
        out = self.proj(out)
        
        # 출력을 원래 형태로 확장
        out = out.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)
        
        return x + out

class TemporalAwareSpatialEncoder(nn.Module):
    """
    시간 및 공간 특성을 고려하는 3D 인코더
    
    Args:
        in_channels (int): 입력 채널 수
        latent_dim (int): 잠재 표현 차원
        base_channels (int): 기본 채널 수
    """
    def __init__(self, in_channels=1, latent_dim=4, base_channels=32):
        super().__init__()
        # 입력 블록
        self.input_block = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 다운샘플링 블록
        self.down1 = nn.Sequential(
            TemporalAwareSpatialBlock(base_channels, base_channels),
            nn.Conv3d(base_channels, base_channels*2, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        )
        
        self.down2 = nn.Sequential(
            TemporalAwareSpatialBlock(base_channels*2, base_channels*2),
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        )
        
        self.down3 = nn.Sequential(
            TemporalAwareSpatialBlock(base_channels*4, base_channels*4),
            nn.Conv3d(base_channels*4, base_channels*8, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        )
        
        # 중간 블록
        self.mid_block = nn.Sequential(
            TemporalAwareSpatialBlock(base_channels*8, base_channels*8),
            TemporalAttention(base_channels*8),
            TemporalAwareSpatialBlock(base_channels*8, base_channels*8)
        )
        
        # 최종 출력
        self.out = nn.Conv3d(base_channels*8, latent_dim, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 비디오 [B, C, T, H, W]
            
        Returns:
            torch.Tensor: 잠재 표현 [B, latent_dim, T, H/8, W/8]
        """
        h = self.input_block(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.mid_block(h)
        latent = self.out(h)
        return latent

class TemporalEncoder(nn.Module):
    """
    시간 차원을 압축하는 인코더
    
    Args:
        in_channels (int): 입력 채널 수
        out_channels (int): 출력 채널 수
    """
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        # 시간적 압축을 위한 컨볼루션 레이어
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0))
        self.norm1 = nn.GroupNorm(8, 64)
        
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(4,1,1), stride=(4,1,1), padding=(0,0,0))
        self.norm2 = nn.GroupNorm(8, 64)
        
        self.conv3 = nn.Conv3d(64, out_channels, kernel_size=(1,1,1))
        
        self.act = nn.SiLU()
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 특징 [B, C, T, H, W]
            
        Returns:
            torch.Tensor: 시간적으로 압축된 특징 [B, out_channels, T/4, H, W]
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        h = self.conv3(h)
        
        return h

class TemporalDecoder(nn.Module):
    """
    시간 차원을 확장하는 디코더
    
    Args:
        in_channels (int): 입력 채널 수
        out_channels (int): 출력 채널 수
    """
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        # 시간적 확장을 위한 역컨볼루션 레이어
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(1,1,1))
        self.norm1 = nn.GroupNorm(8, 64)
        
        self.conv2 = nn.ConvTranspose3d(64, 64, kernel_size=(4,1,1), stride=(4,1,1), padding=(0,0,0))
        self.norm2 = nn.GroupNorm(8, 64)
        
        self.conv3 = nn.Conv3d(64, out_channels, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0))
        
        self.act = nn.SiLU()
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 특징 [B, C, T/4, H, W]
            
        Returns:
            torch.Tensor: 시간적으로 확장된 특징 [B, out_channels, T, H, W]
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        h = self.conv3(h)
        
        return h

class TemporalAwareSpatialDecoder(nn.Module):
    """
    시간 및 공간 특성을 고려하는 3D 디코더
    
    Args:
        latent_dim (int): 잠재 표현 차원
        out_channels (int): 출력 채널 수
        base_channels (int): 기본 채널 수
    """
    def __init__(self, latent_dim=4, out_channels=1, base_channels=32):
        super().__init__()
        # 입력 블록
        self.input_block = nn.Conv3d(latent_dim, base_channels*8, kernel_size=1)
        
        # 중간 블록
        self.mid_block = nn.Sequential(
            TemporalAwareSpatialBlock(base_channels*8, base_channels*8),
            TemporalAttention(base_channels*8),
            TemporalAwareSpatialBlock(base_channels*8, base_channels*8)
        )
        
        # 업샘플링 블록
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
            TemporalAwareSpatialBlock(base_channels*4, base_channels*4)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
            TemporalAwareSpatialBlock(base_channels*2, base_channels*2)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
            TemporalAwareSpatialBlock(base_channels, base_channels)
        )
        
        # 출력 블록
        self.output_block = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 잠재 표현 [B, latent_dim, T, H/8, W/8]
            
        Returns:
            torch.Tensor: 재구성된 비디오 [B, out_channels, T, H, W]
        """
        h = self.input_block(x)
        h = self.mid_block(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        output = self.output_block(h)
        output = torch.sigmoid(output)
        
        return output