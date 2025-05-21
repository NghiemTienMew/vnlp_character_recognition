"""
postprocessor.py - Hậu xử lý kết quả nhận dạng ký tự biển số xe
"""
import re
from typing import Dict, List, Optional, Union
import difflib
from collections import Counter

from config.config import get_config


class Postprocessor:
    """
    Lớp hậu xử lý kết quả nhận dạng biển số xe
    """
    
    def __init__(self):
        """
        Khởi tạo bộ hậu xử lý
        """
        self.config = get_config()
        
        # Áp dụng quy tắc biển số xe
        self.apply_license_plate_rules = self.config.get('postprocessing.apply_license_plate_rules', True)
        
        # Mã tỉnh/thành phố
        self.province_codes = self.config.get('postprocessing.province_codes_set', set())
        
        # Serie ký tự
        self.serie_codes = self.config.get('postprocessing.serie_codes_set', set())
        
        # Sử dụng beam search
        self.use_beam_search = self.config.get('postprocessing.use_beam_search', True)
        self.beam_width = self.config.get('postprocessing.beam_width', 5)
        
        # Sử dụng từ điển
        self.use_lexicon = self.config.get('postprocessing.use_lexicon', True)
        
        # Thiết lập các quy tắc hậu xử lý
        self.setup_rules()
    
    def setup_rules(self):
        """
        Thiết lập các quy tắc hậu xử lý
        """
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
        
        # Mẫu biển số xe Việt Nam
        # Biển số xe cơ bản: nn-XY nnnnn (n: số, X, Y: chữ cái)
        self.plate_patterns = [
            r'^\d{2}[A-Z]{1,2}\-\d{5}$',     # 29A-12345, 29AB-12345
            r'^\d{2}[A-Z]{1}\d{1}\-\d{4}$',   # 29A1-2345
            r'^\d{2}[A-Z]{1}\d{2}\-\d{3}$',   # 29A12-345
            r'^\d{2}[A-Z]{1}\d{3}\-\d{2}$',   # 29A123-45
            r'^\d{2}[A-Z]{1}\d{4}\-\d{1}$'    # 29A1234-5
        ]
    
    def format_license_plate(self, text: str) -> str:
        """
        Định dạng lại chuỗi biển số xe
        
        Args:
            text: Chuỗi biển số xe cần định dạng
            
        Returns:
            str: Chuỗi biển số xe đã định dạng
        """
        # Loại bỏ các ký tự đặc biệt
        text = re.sub(r'[^0-9A-Za-z\-]', '', text)
        
        # Chuyển về chữ hoa
        text = text.upper()
        
        # Nếu không có dấu gạch ngang, thêm vào
        if '-' not in text and len(text) >= 5:
            # Tìm vị trí để thêm dấu gạch ngang
            
            # Trường hợp 1: nnX-nnnnn hoặc nnXY-nnnnn
            match = re.match(r'^(\d{2}[A-Z]{1,2})(\d{5})$', text)
            if match:
                return f"{match.group(1)}-{match.group(2)}"
            
            # Trường hợp 2: nnXn-nnnn
            match = re.match(r'^(\d{2}[A-Z]{1}\d{1})(\d{4})$', text)
            if match:
                return f"{match.group(1)}-{match.group(2)}"
            
            # Trường hợp 3: nnXnn-nnn
            match = re.match(r'^(\d{2}[A-Z]{1}\d{2})(\d{3})$', text)
            if match:
                return f"{match.group(1)}-{match.group(2)}"
            
            # Trường hợp 4: nnXnnn-nn
            match = re.match(r'^(\d{2}[A-Z]{1}\d{3})(\d{2})$', text)
            if match:
                return f"{match.group(1)}-{match.group(2)}"
            
            # Trường hợp 5: nnXnnnn-n
            match = re.match(r'^(\d{2}[A-Z]{1}\d{4})(\d{1})$', text)
            if match:
                return f"{match.group(1)}-{match.group(2)}"
        
        return text
    
    def correct_province_code(self, text: str) -> str:
        """
        Sửa mã tỉnh/thành phố
        
        Args:
            text: Chuỗi biển số xe
            
        Returns:
            str: Chuỗi biển số xe đã sửa mã tỉnh/thành phố
        """
        if not self.apply_license_plate_rules or not text:
            return text
        
        # Lấy 2 ký tự đầu tiên (mã tỉnh)
        if len(text) < 2:
            return text
        
        province_code = text[:2]
        
        # Kiểm tra nếu mã tỉnh hợp lệ
        if province_code.isdigit() and province_code in self.province_codes:
            return text
        
        # Sửa lỗi mã tỉnh dựa vào các cặp nhầm lẫn
        corrected_code = ''
        
        for i, char in enumerate(province_code):
            if not char.isdigit():
                # Thử tìm số tương ứng với ký tự chữ cái
                for digit, chars in self.confusion_pairs.items():
                    if char in chars:
                        corrected_code += digit
                        break
                else:
                    # Nếu không tìm thấy, giữ nguyên
                    corrected_code += char
            else:
                corrected_code += char
        
        # Kiểm tra lại mã tỉnh sau khi sửa
        if corrected_code.isdigit() and corrected_code in self.province_codes:
            return corrected_code + text[2:]
        
        # Nếu không tìm được mã tỉnh hợp lệ, tìm mã tỉnh gần nhất
        if corrected_code.isdigit():
            closest_match = difflib.get_close_matches(corrected_code, list(self.province_codes), n=1)
            if closest_match:
                return closest_match[0] + text[2:]
        
        return text
    
    def correct_serie_code(self, text: str) -> str:
        """
        Sửa mã serie
        
        Args:
            text: Chuỗi biển số xe
            
        Returns:
            str: Chuỗi biển số xe đã sửa mã serie
        """
        if not self.apply_license_plate_rules or not text:
            return text
        
        # Tách các phần của biển số
        parts = text.split('-')
        if len(parts) != 2:
            return text
        
        prefix = parts[0]
        
        # Kiểm tra phần đầu của biển số
        if len(prefix) < 3:
            return text
        
        # Lấy mã serie (ký tự thứ 3 hoặc thứ 3-4)
        serie_start = 2
        serie_end = 3
        
        # Nếu ký tự thứ 3 là chữ cái
        if prefix[2].isalpha():
            # Kiểm tra xem ký tự tiếp theo có phải là chữ cái không
            if len(prefix) > 3 and prefix[3].isalpha():
                serie_end = 4
        else:
            # Nếu ký tự thứ 3 không phải chữ cái, đây không phải biển số hợp lệ
            return text
        
        serie_code = prefix[serie_start:serie_end]
        
        # Kiểm tra từng ký tự trong serie_code
        corrected_serie = ''
        
        for char in serie_code:
            if char.isalpha() and char in self.serie_codes:
                corrected_serie += char
            else:
                # Tìm ký tự chữ cái tương ứng với số
                for letter, digits in self.confusion_pairs.items():
                    if char in digits and letter in self.serie_codes:
                        corrected_serie += letter
                        break
                else:
                    # Nếu không tìm thấy, tìm ký tự gần nhất trong danh sách serie_codes
                    closest_match = difflib.get_close_matches(char, list(self.serie_codes), n=1)
                    if closest_match:
                        corrected_serie += closest_match[0]
                    else:
                        corrected_serie += char
        
        # Thay thế serie_code trong biển số
        corrected_prefix = prefix[:serie_start] + corrected_serie + prefix[serie_end:]
        return corrected_prefix + '-' + parts[1]
    
    def correct_numbers(self, text: str) -> str:
        """
        Sửa các số trong biển số xe
        
        Args:
            text: Chuỗi biển số xe
            
        Returns:
            str: Chuỗi biển số xe đã sửa các số
        """
        if not text:
            return text
        
        # Tách các phần của biển số
        parts = text.split('-')
        if len(parts) != 2:
            return text
        
        prefix, suffix = parts
        
        # Corrected prefix
        corrected_prefix = prefix
        
        # Corrected suffix
        corrected_suffix = ''
        
        for char in suffix:
            if char.isdigit():
                corrected_suffix += char
            else:
                # Tìm số tương ứng với ký tự chữ cái
                for digit, chars in self.confusion_pairs.items():
                    if char in chars and digit.isdigit():
                        corrected_suffix += digit
                        break
                else:
                    # Nếu không tìm thấy, giữ nguyên
                    corrected_suffix += char
        
        return corrected_prefix + '-' + corrected_suffix
    
    def beam_search(self, text: str, beam_width: int = 5) -> str:
        """
        Sử dụng beam search để tìm biển số xe hợp lệ
        
        Args:
            text: Chuỗi biển số xe
            beam_width: Độ rộng của beam
            
        Returns:
            str: Chuỗi biển số xe sau khi áp dụng beam search
        """
        if not self.use_beam_search or not text:
            return text
        
        # Khởi tạo beam
        beam = [(text, 0)]  # (text, score)
        
        # Các bước hậu xử lý
        processing_steps = [
            self.format_license_plate,
            self.correct_province_code,
            self.correct_serie_code,
            self.correct_numbers
        ]
        
        # Thực hiện từng bước hậu xử lý
        for step in processing_steps:
            new_beam = []
            
            for candidate, score in beam:
                # Áp dụng bước hậu xử lý
                processed = step(candidate)
                
                # Nếu kết quả khác với candidate, thêm cả hai vào beam
                if processed != candidate:
                    # Tính điểm cho candidate mới
                    new_score = score
                    
                    # Kiểm tra xem candidate mới có phù hợp với mẫu biển số không
                    for pattern in self.plate_patterns:
                        if re.match(pattern, processed):
                            new_score += 2  # Thêm điểm nếu phù hợp với mẫu
                            break
                    
                    # Kiểm tra mã tỉnh
                    if len(processed) >= 2 and processed[:2] in self.province_codes:
                        new_score += 1
                    
                    # Kiểm tra serie
                    if len(processed) >= 3 and '-' in processed:
                        prefix = processed.split('-')[0]
                        if len(prefix) >= 3 and prefix[2] in self.serie_codes:
                            new_score += 1
                    
                    new_beam.append((processed, new_score))
                
                # Luôn giữ candidate gốc
                new_beam.append((candidate, score))
            
            # Sắp xếp beam theo điểm giảm dần và lấy top beam_width
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Lấy candidate có điểm cao nhất
        return beam[0][0]
    
    def apply_lexicon(self, text: str) -> str:
        """
        Áp dụng từ điển để sửa lỗi
        
        Args:
            text: Chuỗi biển số xe
            
        Returns:
            str: Chuỗi biển số xe sau khi áp dụng từ điển
        """
        if not self.use_lexicon or not self.apply_license_plate_rules or not text:
            return text
        
        # Kiểm tra xem biển số có phù hợp với một trong các mẫu không
        for pattern in self.plate_patterns:
            if re.match(pattern, text):
                return text
        
        # Nếu không phù hợp, tìm biển số gần nhất trong từ điển
        # (trong triển khai thực tế, bạn có thể sử dụng một từ điển thực sự)
        
        # Đây chỉ là cách tiếp cận đơn giản, không sử dụng từ điển thực
        
        # Tạo một biển số hợp lệ dựa trên text hiện tại
        corrected = text
        
        # Nếu không có dấu gạch ngang, thêm vào
        if '-' not in corrected and len(corrected) >= 5:
            corrected = self.format_license_plate(corrected)
        
        # Kiểm tra và sửa mã tỉnh
        corrected = self.correct_province_code(corrected)
        
        # Kiểm tra và sửa mã serie
        corrected = self.correct_serie_code(corrected)
        
        # Kiểm tra và sửa các số
        corrected = self.correct_numbers(corrected)
        
        return corrected
    
    def __call__(self, text: str) -> str:
        """
        Hậu xử lý chuỗi biển số xe
        
        Args:
            text: Chuỗi biển số xe cần hậu xử lý
            
        Returns:
            str: Chuỗi biển số xe đã hậu xử lý
        """
        if not text:
            return text
        
        # Nếu sử dụng beam search
        if self.use_beam_search:
            text = self.beam_search(text, self.beam_width)
        else:
            # Nếu không sử dụng beam search, áp dụng tuần tự các bước
            text = self.format_license_plate(text)
            text = self.correct_province_code(text)
            text = self.correct_serie_code(text)
            text = self.correct_numbers(text)
        
        # Áp dụng từ điển nếu cần
        if self.use_lexicon:
            text = self.apply_lexicon(text)
        
        return text