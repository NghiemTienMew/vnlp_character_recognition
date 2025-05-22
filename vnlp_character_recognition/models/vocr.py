"""
vocr.py - Mô hình nhận dạng ký tự biển số xe (VOCR)
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
    Mô hình nhận dạng ký tự biển số xe (VOCR)
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
        self.num_classes = self.config.get('data.num_classes', 39)  # Số lượng ký tự + special tokens
        self.use_language_model = self.config.get('model.use_language_model', True)
        
        # Spatial Transformer Network
        if self.use_stn:
            self.stn = SpatialTransformer(input_channels=1)
        
        # CNN Backbone
        if self.backbone == 'vgg19':
            self.cnn = VOCRVCNN()
            self.encoder_hidden_size = 256
        elif self.backbone.startswith('efficientnet'):
            self.cnn = VOCREfficientNet(model_name=self.backbone)
            self.encoder_hidden_size = 256  # Đã được điều chỉnh trong VOCREfficientNet
        elif self.backbone.startswith('mobilenet'):
            self.cnn = VOCRMobileNet()
            self.encoder_hidden_size = 256  # Đã được điều chỉnh trong VOCRMobileNet
        elif self.backbone.startswith('resnet'):
            self.cnn = VOCRResNet(model_name=self.backbone)
            self.encoder_hidden_size = 256  # Đã được điều chỉnh trong VOCRResNet
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
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass của mô hình VOCR
        
        Args:
            images: Tensor ảnh đầu vào, kích thước [batch_size, channels, height, width]
            targets: Tensor nhãn, kích thước [batch_size, max_len]
            teacher_forcing_ratio: Tỷ lệ sử dụng teacher forcing
            
        Returns:
            Dict[str, torch.Tensor]: Kết quả dự đoán
                - outputs: Đầu ra của decoder, kích thước [batch_size, max_len, num_classes]
                - encoded_features: Đặc trưng đã mã hóa từ CNN
                - attention_weights: Trọng số attention
                - lm_outputs: Đầu ra của language model (nếu có)
        """
        batch_size = images.size(0)
        
        # Áp dụng Spatial Transformer Network nếu có
        if self.use_stn:
            images = self.stn(images)
        
        # Trích xuất đặc trưng bằng CNN
        encoded_features = self.cnn(images)  # [batch_size, seq_len, encoder_hidden_size]
        
        # Đưa qua decoder
        decoder_outputs = self.decoder(
            encoder_outputs=encoded_features,
            targets=targets,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        outputs = decoder_outputs['outputs']
        attention_weights = decoder_outputs.get('attention_weights')
        
        # Áp dụng language model nếu có
        if self.use_language_model and self.training:
            lm_outputs = self.language_model(outputs)
        else:
            lm_outputs = None
        
        return {
            'outputs': outputs,
            'encoded_features': encoded_features,
            'attention_weights': attention_weights,
            'lm_outputs': lm_outputs
        }
    
    def predict(self, images: torch.Tensor) -> List[str]:
        """
        Dự đoán chuỗi ký tự từ ảnh
        
        Args:
            images: Tensor ảnh đầu vào, kích thước [batch_size, channels, height, width]
            
        Returns:
            List[str]: Danh sách chuỗi ký tự đã dự đoán
        """
        batch_size = images.size(0)
        device = images.device
        
        # Chuyển sang chế độ đánh giá
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(images)
            predictions = outputs['outputs']
            
            # Lấy các ký tự có xác suất cao nhất
            _, predicted_indices = torch.max(predictions, dim=2)  # [batch_size, max_len]
            
            # Chuyển đổi indices thành chuỗi ký tự
            predicted_strings = []
            
            for i in range(batch_size):
                indices = predicted_indices[i].tolist()
                chars = self.config.get('data.idx_to_char')
                
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
                        string.append(chars[idx])
                
                # Kết hợp thành chuỗi
                predicted_string = ''.join(string)
                
                # Áp dụng hậu xử lý nếu có language model
                if self.use_language_model:
                    predicted_string = self.language_model.post_process(predicted_string)
                
                predicted_strings.append(predicted_string)
        
        # Chuyển về chế độ huấn luyện
        self.train()
        
        return predicted_strings