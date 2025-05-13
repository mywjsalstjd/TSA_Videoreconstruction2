# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MovingMNISTDataset(Dataset):
    """
    MovingMNIST 데이터셋 및 잔상 이미지 로더
    
    Args:
        afterimage_path (str): 잔상 이미지 경로, 형태 [N, H, W]
        video_path (str): 비디오 경로, 형태 [T, N, H, W]
        indices (list): 사용할 데이터의 인덱스
    """
    def __init__(self, afterimage_path, video_path, indices):
        super().__init__()
        self.afterimages = np.load(afterimage_path)   # [N, H, W]
        video = np.load(video_path)                   # [T, N, H, W]
        video = np.transpose(video, (1, 0, 2, 3))     # => [N, T, H, W]
        self.videos = video
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            dict:
                - afterimage (torch.Tensor): 잔상 이미지 [1, H, W]
                - video (torch.Tensor): 비디오 [1, T, H, W]
        """
        i = self.indices[idx]
        after = self.afterimages[i]  # [H, W]
        vid   = self.videos[i]       # [T, H, W]
        after = torch.from_numpy(after).float() / 255.
        vid   = torch.from_numpy(vid).float() / 255.
        after = after.unsqueeze(0)   # => [1, H, W]
        vid   = vid.unsqueeze(0)     # => [1, T, H, W]
        return {'afterimage': after, 'video': vid}

def get_data_loaders_no_shuffle(afterimage_path, video_path, batch_size=16, num_workers=2):
    """
    훈련, 검증, 테스트 데이터 로더를 생성합니다. 데이터를 섞지 않고 8:1:1 비율로 순차적으로 분할합니다.
    
    Args:
        afterimage_path (str): 잔상 이미지 경로
        video_path (str): 비디오 경로
        batch_size (int): 배치 크기
        num_workers (int): 데이터 로딩에 사용할 워커 수
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 데이터 수 확인
    data_size = np.load(afterimage_path).shape[0]
    
    # 전체 인덱스에서 8:1:1 비율로 순차적으로 분할 (셔플 없음)
    indices = np.arange(data_size)
    
    train_size = int(data_size * 0.8)
    val_size = int(data_size * 0.1)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # 데이터셋 생성
    train_ds = MovingMNISTDataset(afterimage_path, video_path, train_indices)
    val_ds = MovingMNISTDataset(afterimage_path, video_path, val_indices)
    test_ds = MovingMNISTDataset(afterimage_path, video_path, test_indices)
    
    # 데이터 로더 생성 (모든 로더에서 shuffle=False로 설정)
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=False,  # 훈련 데이터도 섞지 않음
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def get_data_loaders(afterimage_path, video_path, batch_size=16, num_workers=2, 
                     train_indices=None, val_indices=None, test_indices=None):
    """
    훈련, 검증, 테스트 데이터 로더를 생성합니다.
    
    Args:
        afterimage_path (str): 잔상 이미지 경로
        video_path (str): 비디오 경로
        batch_size (int): 배치 크기
        num_workers (int): 데이터 로딩에 사용할 워커 수
        train_indices (list, optional): 훈련 데이터 인덱스
        val_indices (list, optional): 검증 데이터 인덱스
        test_indices (list, optional): 테스트 데이터 인덱스
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 인덱스가 제공되지 않은 경우 기본값 설정
    if train_indices is None or val_indices is None or test_indices is None:
        # 데이터 수 확인
        data_size = np.load(afterimage_path).shape[0]
        
        # 전체 인덱스에서 8:1:1 비율로 분할
        indices = np.arange(data_size)
        np.random.shuffle(indices)
        
        train_size = int(data_size * 0.8)
        val_size = int(data_size * 0.1)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
    
    # 데이터셋 생성
    train_ds = MovingMNISTDataset(afterimage_path, video_path, train_indices)
    val_ds = MovingMNISTDataset(afterimage_path, video_path, val_indices)
    test_ds = MovingMNISTDataset(afterimage_path, video_path, test_indices)
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def check_and_create_dirs(base_dir="datasets"):
    """
    데이터셋 디렉토리가 존재하는지 확인하고 필요한 경우 생성
    
    Args:
        base_dir (str): 기본 데이터셋 디렉토리
        
    Returns:
        tuple: (afterimage_path, video_path) - 데이터셋 경로
    """
    # 데이터셋 디렉토리 확인/생성
    os.makedirs(os.path.join(base_dir, "MovingMNIST"), exist_ok=True)
    
    # 파일 경로
    afterimage_path = os.path.join(base_dir, "MovingMNIST", "mnist_afterimages.npy")
    video_path = os.path.join(base_dir, "MovingMNIST", "mnist_test_seq.npy")
    
    # 상대 경로 사용 옵션 (기본 경로가 없는 경우)
    if not os.path.exists(afterimage_path):
        alternative_afterimage_path = "mnist_afterimages.npy"
        alternative_video_path = "mnist_test_seq.npy"
        
        if os.path.exists(alternative_afterimage_path):
            print(f"Using local paths: {alternative_afterimage_path}, {alternative_video_path}")
            return alternative_afterimage_path, alternative_video_path
        else:
            raise FileNotFoundError(
                f"데이터셋 파일을 찾을 수 없습니다. 경로 확인: {afterimage_path} 또는 {alternative_afterimage_path}"
            )
    
    return afterimage_path, video_path