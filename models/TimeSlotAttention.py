# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import SoftPositionEmbed

class SlotAttention(nn.Module):
    """
    슬롯 어텐션 모듈 - 입력 특징을 슬롯으로 분해
    
    Args:
        num_slots (int): 슬롯 수 (생성할 프레임 수)
        dim (int): 슬롯 및 입력 특징 차원
        iters (int): 슬롯 어텐션 반복 횟수
        eps (float): 안정성을 위한 작은 상수
        hidden_dim (int): MLP의 숨겨진 차원
    """
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots  # 슬롯 수 = 생성할 프레임 수
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        # 슬롯 초기화 파라미터
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        # 어텐션 연산을 위한 레이어
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # 슬롯 업데이트를 위한 GRU
        self.gru = nn.GRUCell(dim, dim)

        # 피드포워드 네트워크
        hidden_dim = max(dim, hidden_dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        # 정규화 레이어
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        """
        Args:
            inputs (torch.Tensor): 입력 특징 [batch_size, num_inputs, dim]
            num_slots (int, optional): 슬롯 수를 재정의 (기본값은 self.num_slots)
            
        Returns:
            tuple:
                - slots (torch.Tensor): 최종 슬롯 [batch_size, num_slots, dim]
                - attn (torch.Tensor): 어텐션 가중치 [batch_size, num_slots, num_inputs]
        """
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        # 슬롯 초기화 (가우시안 노이즈로)
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        # 입력 정규화 및 변환
        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        # 반복적 슬롯 업데이트
        for _ in range(self.iters):
            slots_prev = slots

            # 슬롯 정규화 및 쿼리 생성
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            # 어텐션 계산
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            # 어텐션 기반 업데이트 계산
            updates = torch.einsum('bjd,bij->bid', v, attn)

            # GRU를 통한 슬롯 업데이트
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            # 슬롯 형태 복원 및 피드포워드
            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots, attn

class SlotBasedAfterimageToVideo(nn.Module):
    """
    슬롯 어텐션 기반 잔상→비디오 확장 모듈
    
    잔상 이미지로부터 시간적 슬롯을 추출하고 각 슬롯을 이용해 비디오 프레임을 생성
    
    Args:
        in_channels (int): 입력 채널 수
        out_channels (int): 출력 채널 수
        num_frames (int): 생성할 프레임 수
        hidden_dim (int): 특징 차원
        resolution (tuple): 입력 해상도 (H, W)
        device (torch.device): 모듈이 실행될 장치
    """
    def __init__(self, in_channels=1, out_channels=1, num_frames=20, hidden_dim=64, resolution=(64, 64), device=None):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.resolution = resolution
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 이미지 인코더 - 공간 특징 추출
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU()
        )
        
        # 위치 인코딩 - 공간 정보 주입
        self.pos_encoder = SoftPositionEmbed(hidden_dim, resolution, device=self.device)
        
        # 특징 MLP - 슬롯 어텐션 입력 준비
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 슬롯 어텐션 - 시간적 슬롯으로 분해
        self.slot_attention = SlotAttention(
            num_slots=num_frames,  # 각 슬롯이 하나의 프레임에 해당
            dim=hidden_dim,
            iters=5,  # 더 많은 반복으로 정확한 분해
            hidden_dim=hidden_dim*2
        )
        
        # 시간 순서 인코더 - 각 슬롯에 시간 정보 주입
        self.time_encoder = nn.Embedding(num_frames, hidden_dim)
        
        # 슬롯 프로세서 - 시간 순서대로 정렬
        self.slot_processor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 프레임 생성기 - 공간 특징과 시간 특징 결합하여 프레임 생성
        self.frame_generator = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim*2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim*2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 잔상 이미지 [B, C, H, W]
            
        Returns:
            tuple:
                - video (torch.Tensor): 생성된 비디오 [B, out_channels, num_frames, H, W]
                - attention_masks (torch.Tensor): 어텐션 마스크 [B, 1, num_frames, H, W]
        """
        batch_size = x.shape[0]
        
        # 1. 공간 특징 추출
        spatial_features = self.encoder(x)  # [B, hidden_dim, H, W]
        h, w = spatial_features.shape[2], spatial_features.shape[3]
        
        # 2. 위치 인코딩 및 특징 재구성
        spatial_features = spatial_features.permute(0, 2, 3, 1)  # [B, H, W, hidden_dim]
        spatial_features = self.pos_encoder(spatial_features)
        
        # 3. 특징 평탄화 및 전처리
        flat_features = spatial_features.reshape(batch_size, -1, self.hidden_dim)  # [B, H*W, hidden_dim]
        flat_features = self.mlp(flat_features)
        
        # 4. 슬롯 어텐션으로 시간 슬롯 추출
        slots, attn = self.slot_attention(flat_features)  # [B, num_frames, hidden_dim], [B, num_slots, H*W]
        
        # 5. 시간 정보 주입 (순서대로)
        time_embed = self.time_encoder(torch.arange(self.num_frames, device=x.device))
        time_embed = time_embed.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_frames, hidden_dim]
        slots = slots + time_embed
        
        # 6. 슬롯 처리
        processed_slots = self.slot_processor(slots)  # [B, num_frames, hidden_dim]
        
        # 7. 공간 감응 시계열로 확장
        video_frames = []
        attention_masks = []
        
        # 어텐션 마스크 재구성 (시각화용)
        attn_masks = attn.reshape(batch_size, self.num_frames, h, w)
        
        # 각 슬롯에서 프레임 생성
        for t in range(self.num_frames):
            # 현재 시간 슬롯
            slot_t = processed_slots[:, t].unsqueeze(1)  # [B, 1, hidden_dim]
            
            # 슬롯 특징을 공간적으로 브로드캐스트
            slot_broadcasted = slot_t.reshape(batch_size, self.hidden_dim, 1, 1).expand(-1, -1, h, w)
            
            # 공간 특징
            spatial_feat = spatial_features.permute(0, 3, 1, 2)  # [B, hidden_dim, H, W]
            
            # 공간 특징과 슬롯 특징을 결합
            combined = torch.cat([spatial_feat, slot_broadcasted], dim=1)  # [B, 2*hidden_dim, H, W]
            
            # 프레임 생성
            frame = self.frame_generator(combined)  # [B, out_channels, H, W]
            video_frames.append(frame)
            
            # 어텐션 마스크 저장
            mask_t = attn_masks[:, t].unsqueeze(1)  # [B, 1, H, W]
            attention_masks.append(mask_t)
        
        # 8. 비디오 텐서 조립
        video = torch.stack(video_frames, dim=2)  # [B, out_channels, num_frames, H, W]
        attention_masks = torch.stack(attention_masks, dim=2)  # [B, 1, num_frames, H, W]
        
        return video, attention_masks