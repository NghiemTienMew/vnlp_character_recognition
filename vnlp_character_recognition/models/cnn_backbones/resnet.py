"""
resnet.py - Mạng CNN dựa trên ResNet cho VOCR
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torchvision import models


class VOCRResNet(nn.Module):
    """
    CNN backbone dựa trên ResNet cho VOCR, được tinh chỉnh cho biển số xe
    """
    
    def __init__(self, model_name: str = 'resnet18'):
        """
        Khởi tạo mạng VOCRResNet
        
        Args:
            model_name: Tên của mô hình ResNet ('resnet18', 'resnet34', 'resnet50')
        """
        super(VOCRResNet, self).__init__()
        
        # Tạo model base
        if model_name == 'resnet18':
            base_model = models.resnet18(pretrained=True)
            out_channels = 512
        elif model_name == 'resnet34':
            base_model = models.resnet34(pretrained=True)
            out_channels = 512
        elif model_name == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            out_channels = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Thay thế lớp đầu vào để phù hợp với grayscale (1 channel)
        self.first_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Lấy các features từ model base (bỏ qua lớp đầu vào và fully connected)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        # Thêm các lớp để điều chỉnh kích thước đầu ra
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 70))
        self.final_conv = nn.Conv2d(out_channels, 256, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của VOCRResNet
        
        Args:
            x: Tensor đầu vào, kích thước [batch_size, channels=1, height=32, width=140]
            
        Returns:
            torch.Tensor: Tensor đầu ra, kích thước [batch_size, seq_len=70, hidden_size=256]
        """
        # Lớp đầu vào
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
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