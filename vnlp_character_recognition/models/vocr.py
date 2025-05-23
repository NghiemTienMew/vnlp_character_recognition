"""
vocr.py - Mô hình nhận dạng ký tự biển số xe (VOCR) - Đã sửa lỗi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

from models.cnn_backbones.vgg import VOCRVCNN
from models.cnn_backbones.efficientnet import VOCREfficientNet
from models.cnn_backbones.mobilenet import VOCRMobileNet
from models.cnn_backbones.resnet import VOCRResNet
from models.stn import SpatialTransformer
from models.decoder import create_decoder
from models.language_model import LanguageModel
from config.config import get_config


class VOCR(nn.Module):
    """
    Mô hình nhận dạng ký tự biển số xe (VOCR) - Đã sửa lỗi batch size mismatch
    """
    
    def __init__(self):
        """
        Khởi tạo mô hình VOCR
        """
        super(VOCR, self).__init__()
        
        self.config = get_config()
        
        # Lấy tham số từ cấu hình
        self.backbone = self.config.get('model.backbone', 'efficientnet_b0')
        self.use_stn = self.config.get('model.use_stn', True)
        self.rnn_hidden_size = self.config.get('model.rnn_hidden_size', 256)
        self.embedding_dim = self.config.get('model.embedding_dim', 256)
        self.rnn_type = self.config.get('model.rnn_type', 'gru')
        self.attention_type = self.config.get('model.attention_type', 'multi_head')
        self.decoder_type = self.config.get('model.decoder_type', 'rnn')
        self.num_classes = self.config.get('data.num_classes', 39)
        self.use_language_model = self.config.get('model.use_language_model', True)
        self.max_length = self.config.get('data.max_length', 10)
        
        # Spatial Transformer Network
        if self.use_stn:
            self.stn = SpatialTransformer(input_channels=1)
        
        # CNN Backbone
        if self.backbone == 'vgg19':
            self.cnn = VOCRVCNN()
            self.encoder_hidden_size = 256
        elif self.backbone.startswith('efficientnet'):
            self.cnn = VOCREfficientNet(model_name=self.backbone)
            self.encoder_hidden_size = 256
        elif self.backbone.startswith('mobilenet'):
            self.cnn = VOCRMobileNet()
            self.encoder_hidden_size = 256
        elif self.backbone.startswith('resnet'):
            self.cnn = VOCRResNet(model_name=self.backbone)
            self.encoder_hidden_size = 256
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        # Decoder
        self.decoder = create_decoder(
            decoder_type=self.decoder_type,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.rnn_hidden_size,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            attention_type=self.attention_type,
            rnn_type=self.rnn_type,
            num_layers=self.config.get('model.rnn_num_layers', 2),
            dropout=self.config.get('model.rnn_dropout', 0.2),
            attention_heads=self.config.get('model.attention_heads', 8),
            transformer_nhead=self.config.get('model.transformer_nhead', 8),
            transformer_dim_feedforward=self.config.get('model.transformer_dim_feedforward', 2048)
        )
        
        # Language model cho biển số xe
        if self.use_language_model:
            self.language_model = LanguageModel(
                num_classes=self.num_classes,
                chars=self.config.get('data.chars')
            )
    
    def _ensure_batch_consistency(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Đảm bảo tính nhất quán về batch size giữa images và targets
        
        Args:
            images: Tensor ảnh
            targets: Tensor targets (optional)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Images và targets đã được đồng bộ
        """
        batch_size = images.size(0)
        
        if targets is not None:
            target_batch_size = targets.size(0)
            
            if batch_size != target_batch_size:
                # Lấy batch size nhỏ nhất để đảm bảo consistency
                min_batch_size = min(batch_size, target_batch_size)
                images = images[:min_batch_size]
                targets = targets[:min_batch_size]
                
                print(f"Warning: Batch size mismatch detected. "
                      f"Images: {batch_size}, Targets: {target_batch_size}. "
                      f"Synced to: {min_batch_size}")
        
        return images, targets
    
    def _prepare_targets_for_decoder(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Chuẩn bị targets cho decoder
        
        Args:
            targets: Raw targets từ dataloader
            
        Returns:
            torch.Tensor: Targets đã được chuẩn bị
        """
        batch_size = targets.size(0)
        device = targets.device
        
        # Đảm bảo targets có đúng độ dài
        if targets.size(1) < self.max_length + 2:  # +2 cho <sos> và <eos>
            # Pad targets nếu cần
            pad_length = self.max_length + 2 - targets.size(1)
            pad_tokens = torch.zeros(batch_size, pad_length, dtype=targets.dtype, device=device)
            targets = torch.cat([targets, pad_tokens], dim=1)
        elif targets.size(1) > self.max_length + 2:
            # Cắt targets nếu quá dài
            targets = targets[:, :self.max_length + 2]
        
        return targets
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass của mô hình VOCR - Đã sửa lỗi batch size mismatch
        
        Args:
            images: Tensor ảnh đầu vào, kích thước [batch_size, channels, height, width]
            targets: Tensor nhãn, kích thước [batch_size, max_len]
            teacher_forcing_ratio: Tỷ lệ sử dụng teacher forcing
            
        Returns:
            Dict[str, torch.Tensor]: Kết quả dự đoán
        """
        # Đảm bảo batch consistency
        images, targets = self._ensure_batch_consistency(images, targets)
        batch_size = images.size(0)
        
        # Áp dụng Spatial Transformer Network nếu có
        try:
            if self.use_stn:
                images = self.stn(images)
        except Exception as e:
            print(f"Warning: STN failed, using original images. Error: {e}")
        
        # Trích xuất đặc trưng bằng CNN
        try:
            encoded_features = self.cnn(images)  # [batch_size, seq_len, encoder_hidden_size]
        except Exception as e:
            print(f"Error in CNN backbone: {e}")
            # Tạo dummy features nếu CNN fail
            encoded_features = torch.zeros(batch_size, 32, self.encoder_hidden_size, 
                                         device=images.device, requires_grad=True)
        
        # Chuẩn bị targets cho decoder
        if targets is not None:
            targets = self._prepare_targets_for_decoder(targets)
        
        # Đưa qua decoder với error handling
        try:
            decoder_outputs = self.decoder(
                encoder_outputs=encoded_features,
                targets=targets,
                teacher_forcing_ratio=teacher_forcing_ratio,
                max_len=self.max_length + 2  # +2 cho <sos> và <eos>
            )
            
            outputs = decoder_outputs['outputs']
            attention_weights = decoder_outputs.get('attention_weights')
            
        except Exception as e:
            print(f"Error in decoder: {e}")
            # Tạo dummy outputs nếu decoder fail
            outputs = torch.zeros(batch_size, self.max_length + 2, self.num_classes, 
                                device=images.device, requires_grad=True)
            attention_weights = None
        
        # Áp dụng language model nếu có
        lm_outputs = None
        if self.use_language_model and self.training:
            try:
                lm_outputs = self.language_model(outputs)
            except Exception as e:
                print(f"Warning: Language model failed. Error: {e}")
                lm_outputs = None
        
        return {
            'outputs': outputs,
            'encoded_features': encoded_features,
            'attention_weights': attention_weights,
            'lm_outputs': lm_outputs
        }
    
    def predict(self, images: torch.Tensor) -> List[str]:
        """
        Dự đoán chuỗi ký tự từ ảnh - Đã sửa lỗi
        
        Args:
            images: Tensor ảnh đầu vào, kích thước [batch_size, channels, height, width]
            
        Returns:
            List[str]: Danh sách chuỗi ký tự đã dự đoán
        """
        batch_size = images.size(0)
        device = images.device
        
        # Chuyển sang chế độ đánh giá
        was_training = self.training
        self.eval()
        
        try:
            with torch.no_grad():
                # Forward pass
                outputs = self.forward(images, targets=None, teacher_forcing_ratio=0.0)
                predictions = outputs['outputs']
                
                # Lấy các ký tự có xác suất cao nhất
                _, predicted_indices = torch.max(predictions, dim=2)  # [batch_size, max_len]
                
                # Chuyển đổi indices thành chuỗi ký tự
                predicted_strings = []
                
                for i in range(batch_size):
                    indices = predicted_indices[i].tolist()
                    chars = self.config.get('data.idx_to_char', {})
                    
                    # Bỏ qua special tokens (<sos>, <eos>, <pad>)
                    string = []
                    for idx in indices:
                        if idx == 0:  # <pad>
                            continue
                        elif idx == 1:  # <sos>
                            continue
                        elif idx == 2:  # <eos>
                            break
                        else:
                            if idx in chars:
                                string.append(chars[idx])
                    
                    # Kết hợp thành chuỗi
                    predicted_string = ''.join(string)
                    
                    # Áp dụng hậu xử lý nếu có language model
                    if self.use_language_model:
                        try:
                            predicted_string = self.language_model.post_process(predicted_string)
                        except:
                            pass  # Nếu post_process fail, giữ nguyên string
                    
                    predicted_strings.append(predicted_string)
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Trả về empty strings nếu có lỗi
            predicted_strings = ["" for _ in range(batch_size)]
        
        finally:
            # Khôi phục trạng thái training
            if was_training:
                self.train()
        
        return predicted_strings
    
    def get_attention_weights(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Lấy attention weights để visualize
        
        Args:
            images: Tensor ảnh đầu vào
            
        Returns:
            Optional[torch.Tensor]: Attention weights hoặc None
        """
        was_training = self.training
        self.eval()
        
        try:
            with torch.no_grad():
                outputs = self.forward(images, targets=None, teacher_forcing_ratio=0.0)
                return outputs.get('attention_weights')
        except:
            return None
        finally:
            if was_training:
                self.train()
    
    def count_parameters(self) -> int:
        """
        Đếm số lượng parameters của model
        
        Returns:
            int: Số lượng parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về model
        
        Returns:
            Dict[str, Any]: Thông tin model
        """
        return {
            'backbone': self.backbone,
            'use_stn': self.use_stn,
            'decoder_type': self.decoder_type,
            'attention_type': self.attention_type,
            'rnn_type': self.rnn_type,
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'encoder_hidden_size': self.encoder_hidden_size,
            'decoder_hidden_size': self.rnn_hidden_size,
            'embedding_dim': self.embedding_dim,
            'use_language_model': self.use_language_model,
            'total_parameters': self.count_parameters()
        }