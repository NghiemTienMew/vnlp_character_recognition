"""
efficientnet.py - Mạng CNN dựa trên EfficientNet cho VOCR (đã sửa lỗi)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torchvision import models


class VOCREfficientNet(nn.Module):
    """
    CNN backbone dựa trên EfficientNet cho VOCR, được tinh chỉnh cho biển số xe
    """
    
    def __init__(self, model_name: str = 'efficientnet_b0'):
        """
        Khởi tạo mạng VOCREfficientNet
        
        Args:
            model_name: Tên của mô hình EfficientNet ('efficientnet_b0', 'efficientnet_b1', ...)
        """
        super(VOCREfficientNet, self).__init__()
        
        # Tạo model base với weights thay vì pretrained (để tránh cảnh báo)
        if model_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b1':
            base_model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b2':
            base_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Thay thế lớp đầu vào để phù hợp với grayscale (1 channel)
        self.first_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Lấy các features từ model base (bỏ qua lớp đầu vào và fully connected)
        self.features = nn.Sequential(*list(base_model.features.children())[1:])
        
        # Thêm các lớp để điều chỉnh kích thước đầu ra
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 70))
        self.final_conv = nn.Conv2d(base_model.features[-1][0].out_channels, 256, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của VOCREfficientNet
        
        Args:
            x: Tensor đầu vào, kích thước [batch_size, channels=1, height=32, width=140]
            
        Returns:
            torch.Tensor: Tensor đầu ra, kích thước [batch_size, seq_len=70, hidden_size=256]
        """
        # Lớp đầu vào
        x = self.first_conv(x)
        
        # Đưa qua EfficientNet features
        x = self.features(x)
        
        # Điều chỉnh kích thước không gian
        x = self.adaptive_pool(x)
        
        # Điều chỉnh số kênh
        x = self.final_conv(x)
        
        # Dropout
        x = self.dropout(x)
        
        # x có kích thước [batch_size, channels=256, height=1, width=70]
        # Chuyển đổi để phù hợp với đầu vào của RNN
        x = x.squeeze(2)  # [batch_size, channels=256, width=70]
        x = x.permute(0, 2, 1)  # [batch_size, width=70, channels=256]
        
        return x