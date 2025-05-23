"""
stn.py - Spatial Transformer Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class SpatialTransformer(nn.Module):
    """
    Spatial Transformer Network (STN) giúp học các phép biến đổi hình học
    """
    
    def __init__(self, input_channels: int = 1):
        """
        Khởi tạo Spatial Transformer Network
        
        Args:
            input_channels: Số kênh đầu vào
        """
        super(SpatialTransformer, self).__init__()
        
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Regression network
        # Dự đoán 6 tham số của ma trận biến đổi affine 2D
        # Kích thước đầu vào sau khi qua localization network là 10 * (h/4) * (w/4)
        # Với h=32, w=140 thì sẽ là 10 * 8 * 35 = 2800
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 8 * 35, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        
        # Khởi tạo các tham số của fc_loc để bắt đầu với phép biến đổi đồng nhất
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của STN
        
        Args:
            x: Tensor đầu vào, kích thước [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Tensor đã biến đổi
        """
        batch_size = x.size(0)
        
        # Trích xuất đặc trưng cho localization
        xs = self.localization(x)
        
        # Flatten và đưa qua mạng hồi quy
        xs = xs.view(batch_size, -1)
        theta = self.fc_loc(xs)
        
        # Reshape theta thành ma trận 2x3
        theta = theta.view(batch_size, 2, 3)
        
        # Tạo lưới tọa độ chuẩn
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        
        # Biến đổi ảnh đầu vào
        x = F.grid_sample(x, grid, align_corners=True)
        
        return x