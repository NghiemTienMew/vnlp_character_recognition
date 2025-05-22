"""
mobilenet.py - Mạng CNN dựa trên MobileNet cho VOCR
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torchvision import models


class VOCRMobileNet(nn.Module):
    """
    CNN backbone dựa trên MobileNetV3 cho VOCR, được tinh chỉnh cho biển số xe
    """
    
    def __init__(self, model_name: str = 'mobilenet_v3_small'):
        """
        Khởi tạo mạng VOCRMobileNet
        
        Args:
            model_name: Tên của mô hình MobileNet ('mobilenet_v3_small' hoặc 'mobilenet_v3_large')
        """
        super(VOCRMobileNet, self).__init__()
        
        # Tạo model base
        if model_name == 'mobilenet_v3_small':
            base_model = models.mobilenet_v3_small(pretrained=True)
            out_channels = 576  # Số kênh đầu ra của mạng MobileNetV3-Small
        elif model_name == 'mobilenet_v3_large':
            base_model = models.mobilenet_v3_large(pretrained=True)
            out_channels = 960  # Số kênh đầu ra của mạng MobileNetV3-Large
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Thay thế lớp đầu vào để phù hợp với grayscale (1 channel)
        self.first_conv = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Lấy các features từ model base (bỏ qua lớp đầu vào và fully connected)
        # Đối với MobileNetV3, chúng ta bỏ lớp đầu vào (features[0]) và lớp cuối (features[-1])
        self.features = nn.Sequential(*list(base_model.features.children())[1:-1])
        
        # Thêm các lớp để điều chỉnh kích thước đầu ra
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 70))
        self.final_conv = nn.Conv2d(out_channels, 256, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của VOCRMobileNet
        
        Args:
            x: Tensor đầu vào, kích thước [batch_size, channels=1, height=32, width=140]
            
        Returns:
            torch.Tensor: Tensor đầu ra, kích thước [batch_size, seq_len=70, hidden_size=256]
        """
        # Lớp đầu vào
        x = self.first_conv(x)
        
        # Đưa qua MobileNet features
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