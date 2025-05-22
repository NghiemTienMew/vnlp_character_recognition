"""
combined_loss.py - Kết hợp nhiều loại loss function (đã sửa lỗi signature)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple

from config.config import get_config


class CrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss với label smoothing
    """
    
    def __init__(self, label_smoothing: float = 0.1):
        """
        Khởi tạo Cross Entropy Loss
        
        Args:
            label_smoothing: Hệ số label smoothing
        """
        super(CrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(
        self,
        logits: torch.Tensor,  # [batch_size, seq_len, num_classes]
        targets: torch.Tensor,  # [batch_size, seq_len]
        target_lengths: Optional[torch.Tensor] = None,  # [batch_size]
        lm_outputs: Optional[torch.Tensor] = None  # Không dùng nhưng để tương thích
    ) -> torch.Tensor:
        """
        Forward pass của Cross Entropy Loss
        
        Args:
            logits: Tensor logits, kích thước [batch_size, seq_len, num_classes]
            targets: Tensor targets, kích thước [batch_size, seq_len]
            target_lengths: Độ dài thực của mỗi chuỗi, kích thước [batch_size]
            lm_outputs: Đầu ra của language model (không sử dụng ở đây)
            
        Returns:
            torch.Tensor: Loss
        """
        batch_size, seq_len, num_classes = logits.size()
        
        # Reshape logits cho cross entropy - sử dụng reshape thay vì view
        logits_flat = logits.reshape(-1, num_classes)  # [batch_size * seq_len, num_classes]
        targets_flat = targets.reshape(-1)  # [batch_size * seq_len]
        
        # Tính cross entropy loss
        loss = self.criterion(logits_flat, targets_flat)  # [batch_size * seq_len]
        
        # Reshape lại để phù hợp với kích thước ban đầu
        loss = loss.reshape(batch_size, seq_len)  # [batch_size, seq_len]
        
        # Tạo mask để bỏ qua padding
        mask = (targets != 0).float()  # [batch_size, seq_len]
        
        # Áp dụng mask
        loss = loss * mask  # [batch_size, seq_len]
        
        # Tính loss trung bình
        if target_lengths is not None:
            # Nếu có độ dài thực, tính trung bình theo độ dài thực
            loss_sum = loss.sum(dim=1)  # [batch_size]
            loss = loss_sum / target_lengths.float().clamp(min=1)  # [batch_size]
            loss = loss.mean()
        else:
            # Nếu không có độ dài thực, tính trung bình theo mask
            loss_sum = loss.sum(dim=1)  # [batch_size]
            mask_sum = mask.sum(dim=1)  # [batch_size]
            loss = loss_sum / mask_sum.clamp(min=1)  # [batch_size]
            loss = loss.mean()
        
        return loss


class CTCLoss(nn.Module):
    """
    Connectionist Temporal Classification Loss
    """
    
    def __init__(self):
        """
        Khởi tạo CTC Loss
        """
        super(CTCLoss, self).__init__()
        self.criterion = nn.CTCLoss(zero_infinity=True, reduction='mean')
    
    def forward(
        self,
        logits: torch.Tensor,  # [batch_size, seq_len, num_classes]
        targets: torch.Tensor,  # [batch_size, target_len]
        target_lengths: Optional[torch.Tensor] = None,  # [batch_size]
        lm_outputs: Optional[torch.Tensor] = None  # Không dùng nhưng để tương thích
    ) -> torch.Tensor:
        """
        Forward pass của CTC Loss
        
        Args:
            logits: Tensor logits, kích thước [batch_size, seq_len, num_classes]
            targets: Tensor targets, kích thước [batch_size, target_len]
            target_lengths: Độ dài của targets, kích thước [batch_size]
            lm_outputs: Đầu ra của language model (không sử dụng ở đây)
            
        Returns:
            torch.Tensor: Loss
        """
        batch_size = logits.size(0)
        
        if target_lengths is None:
            # Tính target_lengths từ targets
            target_lengths = torch.zeros(batch_size, dtype=torch.long, device=targets.device)
            for i in range(batch_size):
                for j in range(targets.size(1)):
                    if targets[i, j] == 2:  # <eos>
                        target_lengths[i] = j
                        break
                    if j == targets.size(1) - 1:
                        target_lengths[i] = j + 1
        
        # Tính logits_lengths
        logits_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long, device=logits.device)
        
        # Chuyển đổi định dạng logits sang [seq_len, batch_size, num_classes]
        logits = logits.permute(1, 0, 2)
        
        # Tính CTC loss
        loss = self.criterion(logits, targets, logits_lengths, target_lengths)
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss cho class imbalance
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Khởi tạo Focal Loss
        
        Args:
            alpha: Weight cho class thấp
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        logits: torch.Tensor,  # [batch_size, seq_len, num_classes]
        targets: torch.Tensor,  # [batch_size, seq_len]
        target_lengths: Optional[torch.Tensor] = None,  # [batch_size]
        lm_outputs: Optional[torch.Tensor] = None  # Không dùng nhưng để tương thích
    ) -> torch.Tensor:
        """
        Forward pass của Focal Loss
        
        Args:
            logits: Tensor logits, kích thước [batch_size, seq_len, num_classes]
            targets: Tensor targets, kích thước [batch_size, seq_len]
            target_lengths: Độ dài thực của mỗi chuỗi, kích thước [batch_size]
            lm_outputs: Đầu ra của language model (không sử dụng ở đây)
            
        Returns:
            torch.Tensor: Loss
        """
        batch_size, seq_len, num_classes = logits.size()
        
        # Reshape logits - sử dụng reshape thay vì view
        logits_flat = logits.reshape(-1, num_classes)  # [batch_size * seq_len, num_classes]
        targets_flat = targets.reshape(-1)  # [batch_size * seq_len]
        
        # Tính xác suất bằng softmax
        probs = F.softmax(logits_flat, dim=1)  # [batch_size * seq_len, num_classes]
        
        # Tạo one-hot encoding
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets_flat.unsqueeze(1), 1)
        
        # Tính xác suất của class đúng
        pt = torch.sum(probs * one_hot, dim=1)  # [batch_size * seq_len]
        
        # Tính focal loss
        # Ghi chú: pt không nhỏ hơn epsilon để tránh log(0)
        pt = torch.clamp(pt, 1e-7, 1.0)
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt)  # [batch_size * seq_len]
        
        # Reshape lại
        loss = loss.reshape(batch_size, seq_len)  # [batch_size, seq_len]
        
        # Tạo mask để bỏ qua padding
        mask = (targets != 0).float()  # [batch_size, seq_len]
        
        # Áp dụng mask
        loss = loss * mask  # [batch_size, seq_len]
        
        # Tính loss trung bình
        if target_lengths is not None:
            # Nếu có độ dài thực, tính trung bình theo độ dài thực
            loss_sum = loss.sum(dim=1)  # [batch_size]
            loss = loss_sum / target_lengths.float().clamp(min=1)  # [batch_size]
            loss = loss.mean()
        else:
            # Nếu không có độ dài thực, tính trung bình theo mask
            loss_sum = loss.sum(dim=1)  # [batch_size]
            mask_sum = mask.sum(dim=1)  # [batch_size]
            loss = loss_sum / mask_sum.clamp(min=1)  # [batch_size]
            loss = loss.mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Kết hợp nhiều loại loss function
    """
    
    def __init__(self):
        """
        Khởi tạo Combined Loss
        """
        super(CombinedLoss, self).__init__()
        self.config = get_config()
        
        # Khởi tạo các loss function thành phần
        self.cross_entropy = CrossEntropyLoss(
            label_smoothing=float(self.config.get('loss.label_smoothing', 0.1))
        )
        
        self.ctc_loss = CTCLoss()
        
        self.focal_loss = FocalLoss(
            alpha=float(self.config.get('loss.focal_alpha', 0.25)),
            gamma=float(self.config.get('loss.focal_gamma', 2.0))
        )
        
        # Trọng số cho mỗi loss
        self.cross_entropy_weight = float(self.config.get('loss.cross_entropy_weight', 1.0))
        self.ctc_weight = float(self.config.get('loss.ctc_weight', 0.5))
        self.focal_weight = float(self.config.get('loss.focal_weight', 0.5))
        
        # Loại loss được sử dụng
        self.loss_type = self.config.get('loss.type', 'combined')
        
        # Hard mining
        self.hard_mining = self.config.get('loss.hard_mining', True)
        self.hard_mining_ratio = float(self.config.get('loss.hard_mining_ratio', 0.2))
    
    def _compute_lengths(self, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính độ dài thực của targets và logits
        
        Args:
            targets: Tensor targets, kích thước [batch_size, seq_len]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - targets_lengths: Độ dài của targets
                - logits_lengths: Độ dài của logits
        """
        batch_size = targets.size(0)
        device = targets.device
        
        # Tính độ dài của targets (đến <eos>)
        targets_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            for j in range(targets.size(1)):
                if targets[i, j] == 2:  # <eos>
                    targets_lengths[i] = j + 1
                    break
                if j == targets.size(1) - 1:
                    targets_lengths[i] = j + 1
        
        # Độ dài của logits (thường là max_len)
        logits_lengths = torch.full((batch_size,), targets.size(1), dtype=torch.long, device=device)
        
        return targets_lengths, logits_lengths
    
    def _apply_hard_mining(self, loss: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Áp dụng hard mining (tập trung vào các mẫu khó)
        
        Args:
            loss: Tensor loss, kích thước [batch_size]
            ratio: Tỷ lệ mẫu khó được chọn
            
        Returns:
            torch.Tensor: Loss sau khi áp dụng hard mining
        """
        if not self.hard_mining or ratio >= 1.0:
            return loss.mean()
        
        # Sắp xếp loss giảm dần
        sorted_loss, _ = torch.sort(loss, descending=True)
        
        # Lấy top k% mẫu khó
        k = max(1, int(loss.size(0) * ratio))
        hard_loss = sorted_loss[:k]
        
        return hard_loss.mean()
    
    def forward(
        self,
        logits: torch.Tensor,  # [batch_size, seq_len, num_classes]
        targets: torch.Tensor,  # [batch_size, seq_len]
        target_lengths: Optional[torch.Tensor] = None,  # [batch_size]
        lm_outputs: Optional[torch.Tensor] = None  # [batch_size, seq_len, num_classes]
    ) -> torch.Tensor:
        """
        Forward pass của Combined Loss
        
        Args:
            logits: Tensor logits, kích thước [batch_size, seq_len, num_classes]
            targets: Tensor targets, kích thước [batch_size, seq_len]
            target_lengths: Độ dài thực của targets, kích thước [batch_size]
            lm_outputs: Đầu ra của language model, kích thước [batch_size, seq_len, num_classes]
            
        Returns:
            torch.Tensor: Loss
        """
        batch_size = logits.size(0)
        
        # Sử dụng target_lengths nếu có, nếu không thì tính từ targets
        if target_lengths is None:
            targets_lengths, logits_lengths = self._compute_lengths(targets)
        else:
            targets_lengths = target_lengths
            logits_lengths = torch.full((batch_size,), targets.size(1), dtype=torch.long, device=targets.device)
        
        # Đối với cross entropy và focal loss, cần target từ 1 (bỏ qua <sos>)
        ce_targets = targets[:, 1:]
        ce_logits = logits[:, :-1]
        
        if lm_outputs is not None:
            lm_logits = lm_outputs[:, :-1]
        
        # Kết hợp logits từ model và language model nếu có
        if lm_outputs is not None:
            combined_logits = ce_logits + 0.2 * lm_logits
        else:
            combined_logits = ce_logits
        
        # Tính các loss thành phần
        ce_loss = self.cross_entropy(combined_logits, ce_targets, targets_lengths - 1)
        
        # Nếu sử dụng single loss
        if self.loss_type == 'cross_entropy':
            return ce_loss
        
        elif self.loss_type == 'ctc':
            ctc_loss = self.ctc_loss(logits, targets, targets_lengths, None)
            return ctc_loss
        
        elif self.loss_type == 'focal':
            focal_loss = self.focal_loss(combined_logits, ce_targets, targets_lengths - 1)
            return focal_loss
        
        # Nếu sử dụng combined loss
        else:
            # Tính thêm focal loss nếu cần
            focal_loss = self.focal_loss(combined_logits, ce_targets, targets_lengths - 1) if self.focal_weight > 0 else 0
            
            # Tính thêm CTC loss nếu cần
            ctc_loss = self.ctc_loss(logits, targets, targets_lengths, None) if self.ctc_weight > 0 else 0
            
            # Kết hợp các loss
            total_loss = (
                self.cross_entropy_weight * ce_loss +
                self.ctc_weight * ctc_loss +
                self.focal_weight * focal_loss
            )
            
            return total_loss


def create_loss_function() -> nn.Module:
    """
    Tạo loss function dựa trên cấu hình
    
    Returns:
        nn.Module: Loss function
    """
    config = get_config()
    loss_type = config.get('loss.type', 'combined')
    
    if loss_type == 'cross_entropy':
        return CrossEntropyLoss(
            label_smoothing=float(config.get('loss.label_smoothing', 0.1))
        )
    elif loss_type == 'ctc':
        return CTCLoss()
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=float(config.get('loss.focal_alpha', 0.25)),
            gamma=float(config.get('loss.focal_gamma', 2.0))
        )
    else:  # combined
        return CombinedLoss()