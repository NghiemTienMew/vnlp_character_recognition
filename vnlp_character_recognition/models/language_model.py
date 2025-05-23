"""
language_model.py - Mô hình ngôn ngữ cho biển số xe
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any


class LanguageModel(nn.Module):
    """
    Mô hình ngôn ngữ cho biển số xe, giúp cải thiện độ chính xác
    bằng cách áp dụng các quy tắc biển số xe Việt Nam
    """
    
    def __init__(self, num_classes: int, chars: str):
        """
        Khởi tạo Language Model
        
        Args:
            num_classes: Số lượng lớp (số ký tự)
            chars: Chuỗi ký tự hợp lệ
        """
        super(LanguageModel, self).__init__()
        
        self.num_classes = num_classes
        self.chars = chars
        
        # Mapping giữa ký tự và chỉ số
        self.char_to_idx = {c: i + 3 for i, c in enumerate(chars)}  # +3 cho special tokens
        self.idx_to_char = {i + 3: c for i, c in enumerate(chars)}
        
        # Thêm special tokens
        self.char_to_idx['<pad>'] = 0
        self.char_to_idx['<sos>'] = 1
        self.char_to_idx['<eos>'] = 2
        self.idx_to_char[0] = '<pad>'
        self.idx_to_char[1] = '<sos>'
        self.idx_to_char[2] = '<eos>'
        
        # Thiết lập các quy tắc biển số xe Việt Nam
        self.setup_license_plate_rules()
        
        # Mạng LSTM cho mô hình ngôn ngữ
        self.embedding = nn.Embedding(num_classes, 128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.fc = nn.Linear(512 * 2, num_classes)  # *2 vì bidirectional
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def setup_license_plate_rules(self):
        """
        Thiết lập các quy tắc biển số xe Việt Nam
        """
        # Mã tỉnh/thành phố
        self.province_codes = {
            "11", "12", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
            "30", "31", "32", "33", "34", "35", "36", "37", "38", "40", "41", "42", "43", "47", "48", "49", "50", "51",
            "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
            "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "88",
            "89", "90", "92", "93", "94", "95", "97", "98", "99"
        }
        
        # Serie ký tự
        self.serie_codes = {"A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"}
        
        # Các ký tự dễ bị nhầm lẫn
        self.confusion_pairs = {
            '1': ['I', 'L'],  # Số 1 với chữ I và L  
            '2': ['Z'],       # Số 2 với chữ Z
            '5': ['S'],       # Số 5 với chữ S
            '6': ['G'],       # Số 6 với chữ G
            '8': ['B'],       # Số 8 với chữ B
            '9': ['g', 'q'],  # Số 9 với chữ g, q
            'B': ['8'],       # Chữ B với số 8
            'D': ['0'],       # Chữ D với số 0
            'G': ['6'],       # Chữ G với số 6
            'I': ['1', '7'],  # Chữ I với số 1 và 7
            'O': ['0'],       # Chữ O với số 0
            'S': ['5'],       # Chữ S với số 5
            'Z': ['2']        # Chữ Z với số 2
        }
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của mô hình ngôn ngữ
        
        Args:
            logits: Tensor logits đầu vào, kích thước [batch_size, seq_len, num_classes]
            
        Returns:
            torch.Tensor: Logits đã điều chỉnh bởi mô hình ngôn ngữ
        """
        batch_size, seq_len, _ = logits.size()
        
        # Lấy predicted indices
        _, predicted_indices = torch.max(logits, dim=2)  # [batch_size, seq_len]
        
        # Embedding các indices
        embedded = self.embedding(predicted_indices)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # Đưa qua LSTM
        output, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_size*2]
        output = self.dropout(output)
        
        # Dự đoán lại
        lm_logits = self.fc(output)  # [batch_size, seq_len, num_classes]
        
        # Kết hợp logits ban đầu và logits từ mô hình ngôn ngữ
        combined_logits = logits + 0.2 * lm_logits  # Có thể điều chỉnh hệ số
        
        return combined_logits
    
    def post_process(self, predicted_string: str) -> str:
        """
        Hậu xử lý chuỗi dự đoán dựa trên các quy tắc biển số xe
        
        Args:
            predicted_string: Chuỗi dự đoán ban đầu
            
        Returns:
            str: Chuỗi đã hậu xử lý
        """
        # Bỏ các ký tự không hợp lệ
        cleaned_string = ''.join(c for c in predicted_string if c in self.chars)
        
        # Nếu chuỗi quá ngắn, trả về nguyên bản
        if len(cleaned_string) < 5:
            return cleaned_string
        
        # Xử lý các cặp ký tự dễ nhầm lẫn dựa vào vị trí
        processed_string = list(cleaned_string)
        
        # Tách biển số thành các phần
        if '-' in processed_string:
            # Đã có dấu gạch ngang
            parts = ''.join(processed_string).split('-')
            if len(parts) == 2:
                prefix, suffix = parts
            else:
                # Nếu có nhiều hơn một dấu gạch ngang, giữ nguyên chuỗi
                return ''.join(processed_string)
        else:
            # Chưa có dấu gạch ngang, thử tách
            # Biển số xe Việt Nam thường có dạng: 2 số đầu là mã tỉnh, sau đó là serie (1-2 ký tự), sau đó là 5 số
            
            # Tìm vị trí của ký tự chữ cái đầu tiên
            first_letter_idx = -1
            for i, c in enumerate(processed_string):
                if c.isalpha():
                    first_letter_idx = i
                    break
            
            if first_letter_idx == -1 or first_letter_idx < 2:
                # Không tìm thấy chữ cái hoặc chữ cái ở vị trí không hợp lệ
                return ''.join(processed_string)
            
            # Tìm vị trí của chữ cái cuối cùng liên tiếp
            last_letter_idx = first_letter_idx
            for i in range(first_letter_idx + 1, min(first_letter_idx + 3, len(processed_string))):
                if processed_string[i].isalpha():
                    last_letter_idx = i
                else:
                    break
            
            # Tách biển số
            prefix = ''.join(processed_string[:last_letter_idx + 1])
            suffix = ''.join(processed_string[last_letter_idx + 1:])
        
        # Xử lý phần prefix (mã tỉnh và serie)
        if len(prefix) >= 3:
            # Kiểm tra 2 số đầu là mã tỉnh
            province_code = prefix[:2]
            if not province_code.isdigit():
                # Sửa lỗi nếu có ký tự không phải số
                for i, c in enumerate(province_code):
                    if not c.isdigit():
                        if c in self.confusion_pairs.get('1', []):
                            processed_string[i] = '1'
                        elif c in self.confusion_pairs.get('2', []):
                            processed_string[i] = '2'
                        elif c in self.confusion_pairs.get('5', []):
                            processed_string[i] = '5'
                        elif c in self.confusion_pairs.get('8', []):
                            processed_string[i] = '8'
                        elif c in self.confusion_pairs.get('0', []):
                            processed_string[i] = '0'
            
            # Kiểm tra serie (ký tự sau mã tỉnh)
            for i in range(2, min(4, len(prefix))):
                if i < len(processed_string):
                    c = processed_string[i]
                    if c.isdigit() and int(c) in [1, 2, 5, 6, 8]:
                        # Số có thể bị nhầm với chữ cái
                        possible_letters = self.confusion_pairs.get(c, [])
                        for letter in possible_letters:
                            if letter in self.serie_codes:
                                processed_string[i] = letter
                                break
        
        # Xử lý phần suffix (số thứ tự)
        if suffix:
            for i, c in enumerate(suffix):
                if c.isalpha():
                    # Chữ cái có thể bị nhầm với số
                    if c in self.confusion_pairs:
                        for digit in self.confusion_pairs[c]:
                            if digit.isdigit():
                                processed_string[len(prefix) + 1 + i] = digit
                                break
        
        # Thêm dấu gạch ngang nếu chưa có
        if '-' not in processed_string:
            # Tìm vị trí chữ cái cuối cùng trong phần đầu
            last_letter_idx = -1
            for i, c in enumerate(processed_string):
                if c.isalpha():
                    if i > last_letter_idx:
                        last_letter_idx = i
                    if i > 3:  # Chỉ xét đến vị trí thứ 4
                        break
            
            if last_letter_idx > 0:
                processed_string.insert(last_letter_idx + 1, '-')
        
        return ''.join(processed_string)
    
    def correct_confusion_pairs(self, string: str, position_type: str) -> str:
        """
        Sửa lỗi các ký tự dễ nhầm lẫn dựa vào vị trí
        
        Args:
            string: Chuỗi cần sửa
            position_type: Loại vị trí ('province_code', 'serie', 'number')
            
        Returns:
            str: Chuỗi đã sửa
        """
        result = list(string)
        
        if position_type == 'province_code':
            # Mã tỉnh chỉ có số
            for i, c in enumerate(result):
                if not c.isdigit():
                    for digit, letters in self.confusion_pairs.items():
                        if digit.isdigit() and c in letters:
                            result[i] = digit
                            break
        
        elif position_type == 'serie':
            # Serie chỉ có chữ cái
            for i, c in enumerate(result):
                if c.isdigit():
                    for letter, digits in self.confusion_pairs.items():
                        if letter.isalpha() and c in digits and letter in self.serie_codes:
                            result[i] = letter
                            break
        
        elif position_type == 'number':
            # Phần số chỉ có số
            for i, c in enumerate(result):
                if not c.isdigit():
                    for digit, letters in self.confusion_pairs.items():
                        if digit.isdigit() and c in letters:
                            result[i] = digit
                            break
        
        return ''.join(result)