"""
vgg.py - Mạng CNN dựa trên VGG19 cho VOCR
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class VOCRVCNN(nn.Module):
    """
    CNN backbone dựa trên VGG19 cho VOCR, được tinh chỉnh cho biển số xe
    """
    
    def __init__(self):
        """
        Khởi tạo mạng VOCR-CNN
        """
        super(VOCRVCNN, self).__init__()
        
        # Lớp 1
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 32x140 -> 16x70
        
        # Lớp 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x70 -> 8x35
        
        # Lớp 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))  # 8x35 -> 4x35
        
        # Lớp 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_4 = nn.BatchNorm2d(512)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))  # 4x35 -> 2x35
        
        # Lớp 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_4 = nn.BatchNorm2d(512)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))  # 2x35 -> 1x35
        
        # Lớp cuối
        self.conv6 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của VOCR-CNN
        
        Args:
            x: Tensor đầu vào, kích thước [batch_size, channels=1, height=32, width=140]
            
        Returns:
            torch.Tensor: Tensor đầu ra, kích thước [batch_size, seq_len=70, hidden_size=256]
        """
        # Lớp 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        # Lớp 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        # Lớp 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = F.relu(self.bn3_4(self.conv3_4(x)))
        x = self.pool3(x)
        
        # Lớp 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        x = F.relu(self.bn4_4(self.conv4_4(x)))
        x = self.pool4(x)
        
        # Lớp 5
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        x = F.relu(self.bn5_4(self.conv5_4(x)))
        x = self.pool5(x)
        
        # Lớp cuối
        x = F.relu(self.conv6(x))
        
        # Dropout
        x = self.dropout(x)
        
        # x có kích thước [batch_size, channels=256, height=1, width=70]
        # Chuyển đổi để phù hợp với đầu vào của RNN
        x = x.squeeze(2)  # [batch_size, channels=256, width=70]
        x = x.permute(0, 2, 1)  # [batch_size, width=70, channels=256]
        
        return x