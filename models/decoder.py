"""
decoder.py - Decoder cho mô hình nhận dạng ký tự
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union
import math

from models.attention import create_attention_mechanism


class RNNDecoder(nn.Module):
    """
    RNN Decoder cho mô hình nhận dạng ký tự
    """
    
    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        embedding_dim: int,
        num_classes: int,
        attention_type: str = 'multi_head',
        rnn_type: str = 'gru',
        num_layers: int = 1,
        dropout: float = 0.1,
        attention_heads: int = 8
    ):
        """
        Khởi tạo RNN Decoder
        
        Args:
            encoder_hidden_size: Kích thước ẩn của encoder
            decoder_hidden_size: Kích thước ẩn của decoder
            embedding_dim: Kích thước của embedding
            num_classes: Số lượng lớp (số ký tự)
            attention_type: Loại attention ('additive' hoặc 'multi_head')
            rnn_type: Loại RNN ('gru' hoặc 'lstm')
            num_layers: Số lớp RNN
            dropout: Tỷ lệ dropout
            attention_heads: Số đầu attention (nếu dùng multi-head)
        """
        super(RNNDecoder, self).__init__()
        
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.attention_type = attention_type
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Embedding layer
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Attention mechanism
        if self.attention_type == 'multi_head':
            model_dim = encoder_hidden_size  # Đối với multi-head, model_dim = encoder_hidden_size
            self.attention = create_attention_mechanism(
                attention_type, 
                decoder_hidden_size, 
                encoder_hidden_size,
                model_dim,
                attention_heads
            )
            self.combine_context = nn.Linear(model_dim + embedding_dim, embedding_dim)
        else:
            self.attention = create_attention_mechanism(
                attention_type, 
                decoder_hidden_size, 
                encoder_hidden_size
            )
            self.combine_context = nn.Linear(encoder_hidden_size + embedding_dim, embedding_dim)
        
        # RNN layer
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=decoder_hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=decoder_hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Output layer
        self.output_layer = nn.Linear(decoder_hidden_size + embedding_dim + encoder_hidden_size, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def init_hidden(self, batch_size: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Khởi tạo trạng thái ẩn ban đầu của decoder
        
        Args:
            batch_size: Kích thước batch
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                - Trạng thái ẩn ban đầu (GRU)
                - (Trạng thái ẩn ban đầu, trạng thái cell ban đầu) (LSTM)
        """
        device = next(self.parameters()).device
        
        if self.rnn_type == 'gru':
            return torch.zeros(self.num_layers, batch_size, self.decoder_hidden_size, device=device)
        else:  # lstm
            return (
                torch.zeros(self.num_layers, batch_size, self.decoder_hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.decoder_hidden_size, device=device)
            )
    
    def forward_step(
        self,
        input_char: torch.Tensor,  # [batch_size, 1]
        hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        encoder_outputs: torch.Tensor,  # [batch_size, seq_len, encoder_hidden_size]
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Một bước decoder
        
        Args:
            input_char: Ký tự đầu vào, kích thước [batch_size, 1]
            hidden: Trạng thái ẩn của RNN
            encoder_outputs: Đầu ra từ encoder, kích thước [batch_size, seq_len, encoder_hidden_size]
            encoder_mask: Mask để che các vị trí padding trong encoder outputs
            
        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
                - output: Đầu ra của decoder, kích thước [batch_size, num_classes]
                - hidden: Trạng thái ẩn mới
                - attention_weights: Trọng số attention
        """
        # Embedding input character
        embedded = self.embedding(input_char)  # [batch_size, 1, embedding_dim]
        embedded = self.dropout(embedded)
        
        # Lấy trạng thái ẩn hiện tại cho tính toán attention
        if self.rnn_type == 'gru':
            current_hidden = hidden[-1].unsqueeze(0)  # [1, batch_size, decoder_hidden_size]
        else:  # lstm
            current_hidden = hidden[0][-1].unsqueeze(0)  # [1, batch_size, decoder_hidden_size]
        
        # Áp dụng attention
        if self.attention_type == 'multi_head':
            # Đối với multi-head attention, query cần có kích thước [batch_size, query_len, model_dim]
            # Ở đây query_len = 1
            query = current_hidden.transpose(0, 1)  # [batch_size, 1, decoder_hidden_size]
            
            # Tạo mask cho multi-head attention nếu cần
            if encoder_mask is not None:
                # Chuyển đổi mask để phù hợp với định dạng [batch_size, 1, 1, seq_len]
                mha_mask = encoder_mask.unsqueeze(1).unsqueeze(2)
            else:
                mha_mask = None
            
            context, attention_weights = self.attention(
                query, encoder_outputs, encoder_outputs, mha_mask
            )
            context = context.squeeze(1)  # [batch_size, encoder_hidden_size]
        else:  # additive attention
            # Đối với additive attention, query có kích thước [batch_size, decoder_hidden_size]
            query = current_hidden.squeeze(0)  # [batch_size, decoder_hidden_size]
            
            context, attention_weights = self.attention(
                query, encoder_outputs, encoder_outputs, encoder_mask
            )
        
        # Kết hợp context vector với embedding
        embedded = embedded.squeeze(1)  # [batch_size, embedding_dim]
        combined_input = self.combine_context(torch.cat([context, embedded], dim=1))
        combined_input = torch.tanh(combined_input)
        combined_input = combined_input.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # Đưa qua RNN
        if self.rnn_type == 'gru':
            output, hidden = self.rnn(combined_input, hidden)
        else:  # lstm
            output, hidden = self.rnn(combined_input, hidden)
        
        # Kết hợp đầu ra RNN, context vector và embedding cho dự đoán
        output = output.squeeze(1)  # [batch_size, decoder_hidden_size]
        embedded = embedded  # [batch_size, embedding_dim]
        context = context  # [batch_size, encoder_hidden_size]
        
        output = self.output_layer(torch.cat([output, embedded, context], dim=1))
        
        return output, hidden, attention_weights
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,  # [batch_size, seq_len, encoder_hidden_size]
        targets: Optional[torch.Tensor] = None,  # [batch_size, max_len]
        teacher_forcing_ratio: float = 0.5,
        max_len: Optional[int] = None,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass của decoder
        
        Args:
            encoder_outputs: Đầu ra từ encoder, kích thước [batch_size, seq_len, encoder_hidden_size]
            targets: Nhãn cho teacher forcing, kích thước [batch_size, max_len]
            teacher_forcing_ratio: Tỷ lệ sử dụng teacher forcing
            max_len: Độ dài tối đa của chuỗi đầu ra
            encoder_mask: Mask để che các vị trí padding trong encoder outputs
            
        Returns:
            Dict[str, torch.Tensor]:
                - outputs: Đầu ra của decoder, kích thước [batch_size, max_len, num_classes]
                - attention_weights: Trọng số attention, kích thước [batch_size, max_len, seq_len]
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Xác định độ dài tối đa
        if max_len is None:
            max_len = 20 if targets is None else targets.size(1)
        
        # Khởi tạo trạng thái ẩn
        hidden = self.init_hidden(batch_size)
        
        # Khởi tạo ký tự đầu vào đầu tiên (<sos>)
        input_char = torch.zeros(batch_size, 1, dtype=torch.long, device=device)  # [batch_size, 1]
        
        # Khởi tạo danh sách để lưu outputs và attention weights
        outputs = []
        attention_weights_list = []
        
        # Vòng lặp qua mỗi vị trí trong chuỗi đầu ra
        for t in range(max_len):
            # Một bước decoder
            output, hidden, attention_weights = self.forward_step(
                input_char, hidden, encoder_outputs, encoder_mask
            )
            
            # Lưu output và attention weights
            outputs.append(output)
            attention_weights_list.append(attention_weights)
            
            # Quyết định ký tự đầu vào tiếp theo (teacher forcing hoặc dự đoán)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            if teacher_force and targets is not None:
                input_char = targets[:, t].unsqueeze(1)  # [batch_size, 1]
            else:
                # Dự đoán ký tự có xác suất cao nhất
                top1 = output.argmax(1)
                input_char = top1.unsqueeze(1)  # [batch_size, 1]
            
            # Kiểm tra nếu tất cả các ký tự là <eos>, dừng vòng lặp
            if (input_char == 2).all():  # Giả sử <eos> có index là 2
                break
        
        # Stack các outputs và attention weights
        outputs = torch.stack(outputs, dim=1)  # [batch_size, max_len, num_classes]
        attention_weights = torch.stack(attention_weights_list, dim=1)  # [batch_size, max_len, seq_len]
        
        return {
            'outputs': outputs,
            'attention_weights': attention_weights
        }


class PositionalEncoding(nn.Module):
    """
    Positional Encoding cho Transformer
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Khởi tạo Positional Encoding
        
        Args:
            d_model: Kích thước của model
            dropout: Tỷ lệ dropout
            max_len: Độ dài tối đa của chuỗi
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của Positional Encoding
        
        Args:
            x: Tensor đầu vào, kích thước [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Tensor đầu ra với positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder cho mô hình nhận dạng ký tự
    """
    
    def __init__(
        self,
        encoder_hidden_size: int,
        embedding_dim: int,
        num_classes: int,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        """
        Khởi tạo Transformer Decoder
        
        Args:
            encoder_hidden_size: Kích thước ẩn của encoder
            embedding_dim: Kích thước của embedding
            num_classes: Số lượng lớp (số ký tự)
            nhead: Số đầu attention
            dim_feedforward: Kích thước của feedforward network
            num_layers: Số lớp decoder
            dropout: Tỷ lệ dropout
        """
        super(TransformerDecoder, self).__init__()
        
        self.encoder_hidden_size = encoder_hidden_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Embedding layer
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        # Chuyển đổi kích thước nếu encoder_hidden_size khác embedding_dim
        self.input_projection = nn.Linear(encoder_hidden_size, embedding_dim) if encoder_hidden_size != embedding_dim else None
        
        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, num_classes)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Tạo mask để ngăn chặn attention đến các vị trí tương lai
        
        Args:
            sz: Kích thước của mask
            
        Returns:
            torch.Tensor: Mask, kích thước [sz, sz]
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,  # [batch_size, seq_len, encoder_hidden_size]
        targets: Optional[torch.Tensor] = None,  # [batch_size, max_len]
        teacher_forcing_ratio: float = 1.0,  # Transformer thường dùng teacher forcing hoàn toàn khi huấn luyện
        max_len: Optional[int] = None,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass của Transformer decoder
        
        Args:
            encoder_outputs: Đầu ra từ encoder, kích thước [batch_size, seq_len, encoder_hidden_size]
            targets: Nhãn cho teacher forcing, kích thước [batch_size, max_len]
            teacher_forcing_ratio: Tỷ lệ sử dụng teacher forcing (thường là 1.0 cho Transformer)
            max_len: Độ dài tối đa của chuỗi đầu ra
            encoder_mask: Mask để che các vị trí padding trong encoder outputs
            
        Returns:
            Dict[str, torch.Tensor]:
                - outputs: Đầu ra của decoder, kích thước [batch_size, max_len, num_classes]
                - attention_weights: Không có trong transformer decoder
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Xác định độ dài tối đa
        if max_len is None:
            max_len = 20 if targets is None else targets.size(1)
        
        # Nếu cần, chuyển đổi kích thước của encoder outputs
        if self.input_projection is not None:
            memory = self.input_projection(encoder_outputs)
        else:
            memory = encoder_outputs
        
        # Tạo memory mask nếu cần
        if encoder_mask is not None:
            memory_key_padding_mask = ~encoder_mask.bool()  # Đảo ngược mask
        else:
            memory_key_padding_mask = None
        
        if self.training and targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # Teacher forcing mode (thường dùng trong huấn luyện)
            
            # Embedding các ký tự target
            target_embedding = self.embedding(targets)  # [batch_size, max_len, embedding_dim]
            
            # Thêm positional encoding
            target_embedding = self.pos_encoder(target_embedding)
            
            # Tạo tgt mask để ngăn chặn attention đến các vị trí tương lai
            tgt_mask = self.generate_square_subsequent_mask(targets.size(1)).to(device)
            
            # Tạo tgt key padding mask
            tgt_key_padding_mask = (targets == 0)  # Giả sử 0 là <pad>
            
            # Forward qua transformer decoder
            output = self.transformer_decoder(
                tgt=target_embedding,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # Linear layer để dự đoán ký tự tiếp theo
            outputs = self.output_layer(output)
            
            return {
                'outputs': outputs,
                'attention_weights': None  # Transformer không trả về attention weights trực tiếp
            }
        else:
            # Autoregressive mode (thường dùng trong inference)
            
            # Khởi tạo đầu ra
            outputs = []
            
            # Khởi tạo ký tự đầu vào đầu tiên (<sos>)
            input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)  # [batch_size, 1]
            
            # Vòng lặp qua mỗi vị trí trong chuỗi đầu ra
            for t in range(max_len):
                # Embedding ký tự đầu vào
                input_embedding = self.embedding(input_token)  # [batch_size, t+1, embedding_dim]
                
                # Thêm positional encoding
                input_embedding = self.pos_encoder(input_embedding)
                
                # Tạo tgt mask
                tgt_mask = self.generate_square_subsequent_mask(input_embedding.size(1)).to(device)
                
                # Forward qua transformer decoder
                output = self.transformer_decoder(
                    tgt=input_embedding,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
                
                # Chỉ lấy đầu ra của token cuối cùng
                last_output = output[:, -1]  # [batch_size, embedding_dim]
                
                # Linear layer để dự đoán ký tự tiếp theo
                pred = self.output_layer(last_output)  # [batch_size, num_classes]
                
                # Lưu dự đoán
                outputs.append(pred)
                
                # Quyết định ký tự đầu vào tiếp theo
                if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    next_token = targets[:, t].unsqueeze(1)  # [batch_size, 1]
                else:
                    next_token = pred.argmax(dim=1, keepdim=True)  # [batch_size, 1]
                
                # Nối token mới vào chuỗi đầu vào
                input_token = torch.cat([input_token, next_token], dim=1)  # [batch_size, t+2]
                
                # Kiểm tra nếu tất cả các ký tự là <eos>, dừng vòng lặp
                if (next_token == 2).all():  # Giả sử <eos> có index là 2
                    break
            
            # Stack các outputs
            outputs = torch.stack(outputs, dim=1)  # [batch_size, len, num_classes]
            
            return {
                'outputs': outputs,
                'attention_weights': None  # Transformer không trả về attention weights trực tiếp
            }


def create_decoder(
    decoder_type: str,
    encoder_hidden_size: int,
    decoder_hidden_size: int,
    embedding_dim: int,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Tạo decoder dựa trên loại được chỉ định
    
    Args:
        decoder_type: Loại decoder ('rnn' hoặc 'transformer')
        encoder_hidden_size: Kích thước ẩn của encoder
        decoder_hidden_size: Kích thước ẩn của decoder
        embedding_dim: Kích thước của embedding
        num_classes: Số lượng lớp (số ký tự)
        **kwargs: Tham số bổ sung
        
    Returns:
        nn.Module: Module decoder
    """
    if decoder_type == 'rnn':
        return RNNDecoder(
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            attention_type=kwargs.get('attention_type', 'multi_head'),
            rnn_type=kwargs.get('rnn_type', 'gru'),
            num_layers=kwargs.get('num_layers', 1),
            dropout=kwargs.get('dropout', 0.1),
            attention_heads=kwargs.get('attention_heads', 8)
        )
    elif decoder_type == 'transformer':
        return TransformerDecoder(
            encoder_hidden_size=encoder_hidden_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            nhead=kwargs.get('transformer_nhead', 8),
            dim_feedforward=kwargs.get('transformer_dim_feedforward', 2048),
            num_layers=kwargs.get('num_layers', 6),
            dropout=kwargs.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unsupported decoder type: {decoder_type}")