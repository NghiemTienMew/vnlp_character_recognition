"""
attention.py - Các cơ chế attention cho mô hình nhận dạng ký tự (đã sửa lỗi)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Dict, Optional, Union


class AdditiveAttention(nn.Module):
    """
    Cơ chế Additive Attention (Bahdanau)
    v * tanh(W_q * Q + W_k * K)
    """
    
    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int = 256):
        """
        Khởi tạo Additive Attention
        
        Args:
            query_dim: Kích thước của vector query
            key_dim: Kích thước của vector key
            hidden_dim: Kích thước ẩn trong mạng attention
        """
        super(AdditiveAttention, self).__init__()
        self.query_layer = nn.Linear(query_dim, hidden_dim, bias=False)
        self.key_layer = nn.Linear(key_dim, hidden_dim, bias=False)
        self.energy_layer = nn.Linear(hidden_dim, 1, bias=False)
        
        # Khởi tạo các trọng số
        nn.init.xavier_uniform_(self.query_layer.weight)
        nn.init.xavier_uniform_(self.key_layer.weight)
        nn.init.xavier_uniform_(self.energy_layer.weight)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass của Additive Attention
        
        Args:
            query: Tensor query, kích thước [batch_size, query_dim]
            key: Tensor key, kích thước [batch_size, seq_len, key_dim]
            value: Tensor value, kích thước [batch_size, seq_len, value_dim]
            mask: Tensor mask để che các vị trí padding, kích thước [batch_size, seq_len]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - context vector, kích thước [batch_size, value_dim]
                - attention weights, kích thước [batch_size, seq_len]
        """
        batch_size, seq_len, _ = key.size()
        
        # Mở rộng query để phù hợp với kích thước của key
        query = query.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, query_dim]
        
        # Tính toán hidden representation
        query_hidden = self.query_layer(query)  # [batch_size, seq_len, hidden_dim]
        key_hidden = self.key_layer(key)  # [batch_size, seq_len, hidden_dim]
        
        # Tính toán năng lượng
        energy = self.energy_layer(torch.tanh(query_hidden + key_hidden))  # [batch_size, seq_len, 1]
        energy = energy.squeeze(2)  # [batch_size, seq_len]
        
        # Áp dụng mask nếu có
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # Tính trọng số attention bằng softmax
        attention_weights = F.softmax(energy, dim=1)  # [batch_size, seq_len]
        
        # Tính context vector
        context = torch.bmm(attention_weights.unsqueeze(1), value).squeeze(1)  # [batch_size, value_dim]
        
        return context, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Cơ chế Multi-Head Attention
    """
    
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Khởi tạo Multi-Head Attention
        
        Args:
            model_dim: Kích thước của model
            num_heads: Số lượng đầu attention
            dropout: Tỷ lệ dropout
        """
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        # Linear layers cho query, key, value
        self.query_layer = nn.Linear(model_dim, model_dim)
        self.key_layer = nn.Linear(model_dim, model_dim)
        self.value_layer = nn.Linear(model_dim, model_dim)
        
        # Output projection
        self.output_layer = nn.Linear(model_dim, model_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Khởi tạo các trọng số
        nn.init.xavier_uniform_(self.query_layer.weight)
        nn.init.xavier_uniform_(self.key_layer.weight)
        nn.init.xavier_uniform_(self.value_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chia tensor thành nhiều đầu
        
        Args:
            x: Tensor đầu vào, kích thước [batch_size, seq_len, model_dim]
            
        Returns:
            torch.Tensor: Tensor đã chia, kích thước [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.size()
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gộp các đầu lại với nhau
        
        Args:
            x: Tensor đầu vào, kích thước [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            torch.Tensor: Tensor đã gộp, kích thước [batch_size, seq_len, model_dim]
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.model_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass của Multi-Head Attention
        
        Args:
            query: Tensor query, kích thước [batch_size, query_len, model_dim]
            key: Tensor key, kích thước [batch_size, key_len, model_dim]
            value: Tensor value, kích thước [batch_size, key_len, model_dim]
            mask: Tensor mask để che các vị trí padding, kích thước [batch_size, 1, 1, key_len]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output, kích thước [batch_size, query_len, model_dim]
                - attention weights, kích thước [batch_size, num_heads, query_len, key_len]
        """
        batch_size = query.size(0)
        
        # Linear projections
        query = self.query_layer(query)  # [batch_size, query_len, model_dim]
        key = self.key_layer(key)  # [batch_size, key_len, model_dim]
        value = self.value_layer(value)  # [batch_size, key_len, model_dim]
        
        # Chia thành nhiều đầu
        query = self.split_heads(query)  # [batch_size, num_heads, query_len, head_dim]
        key = self.split_heads(key)  # [batch_size, num_heads, key_len, head_dim]
        value = self.split_heads(value)  # [batch_size, num_heads, key_len, head_dim]
        
        # Scaled dot-product attention
        # Tính toán điểm tích có thể chia
        scores = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, num_heads, query_len, key_len]
        scores = scores / math.sqrt(self.head_dim)
        
        # Áp dụng mask nếu có
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        
        # Tính trọng số attention bằng softmax
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, query_len, key_len]
        attention_weights = self.dropout(attention_weights)
        
        # Áp dụng trọng số vào value
        context = torch.matmul(attention_weights, value)  # [batch_size, num_heads, query_len, head_dim]
        
        # Gộp các đầu lại với nhau
        context = self.combine_heads(context)  # [batch_size, query_len, model_dim]
        
        # Linear projection cuối cùng
        output = self.output_layer(context)  # [batch_size, query_len, model_dim]
        
        return output, attention_weights


def create_attention_mechanism(attention_type: str, query_dim: int, key_dim: int, 
                               model_dim: int = 256, num_heads: int = 8) -> nn.Module:
    """
    Tạo cơ chế attention dựa trên loại được chỉ định
    
    Args:
        attention_type: Loại attention ('additive' hoặc 'multi_head')
        query_dim: Kích thước của vector query
        key_dim: Kích thước của vector key
        model_dim: Kích thước của model (cho multi-head attention)
        num_heads: Số lượng đầu (cho multi-head attention)
        
    Returns:
        nn.Module: Module attention
    """
    if attention_type == 'additive':
        return AdditiveAttention(query_dim, key_dim, hidden_dim=model_dim)
    elif attention_type == 'multi_head':
        return MultiHeadAttention(model_dim, num_heads)
    else:
        raise ValueError(f"Unsupported attention type: {attention_type}")