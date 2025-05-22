"""
preprocessor.py - Các phương pháp tiền xử lý ảnh biển số xe
"""
import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import Union, Tuple, List, Optional
from config.config import get_config


class Preprocessor:
    """
    Lớp tiền xử lý ảnh biển số xe
    """
    
    def __init__(self):
        """
        Khởi tạo bộ tiền xử lý ảnh
        """
        self.config = get_config()
        
        # Kích thước ảnh đầu vào cho mô hình
        self.image_size = self.config.get('data.image_size', (32, 140))
        
        # Tham số chuẩn hóa
        self.normalize_mean = self.config.get('preprocessing.normalize_mean', [0.5])
        self.normalize_std = self.config.get('preprocessing.normalize_std', [0.5])
        
        # Tham số CLAHE
        self.use_clahe = self.config.get('preprocessing.use_clahe', True)
        self.clip_limit = self.config.get('preprocessing.clip_limit', 2.0)
        self.tile_grid_size = self.config.get('preprocessing.tile_grid_size', (8, 8))
        
        # Tham số ngưỡng thích ứng
        self.adaptive_threshold = self.config.get('preprocessing.adaptive_threshold', True)
        
        # Chuẩn bị bộ chuyển đổi tensor
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(self.normalize_mean, self.normalize_std)
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Thay đổi kích thước ảnh theo kích thước đầu vào của mô hình
        
        Args:
            image: Ảnh đầu vào dạng numpy array
            
        Returns:
            np.ndarray: Ảnh đã thay đổi kích thước
        """
        return cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Ảnh đầu vào dạng numpy array
            
        Returns:
            np.ndarray: Ảnh đã áp dụng CLAHE
        """
        if not self.use_clahe:
            return image
        
        # Tạo đối tượng CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        
        # Áp dụng CLAHE
        return clahe.apply(image)
    
    def apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Áp dụng ngưỡng thích ứng để phân tách ký tự và nền
        
        Args:
            image: Ảnh đầu vào dạng numpy array
            
        Returns:
            np.ndarray: Ảnh đã áp dụng ngưỡng
        """
        if not self.adaptive_threshold:
            return image
        
        # Áp dụng ngưỡng thích ứng
        binary_image = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        return binary_image
    
    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        Điều chỉnh góc nghiêng của biển số
        
        Args:
            image: Ảnh đầu vào dạng numpy array
            
        Returns:
            np.ndarray: Ảnh đã điều chỉnh góc nghiêng
        """
        # Tìm các cạnh trong ảnh
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Sử dụng Hough Line Transform để tìm các đường thẳng
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        # Nếu không tìm thấy đường thẳng, trả về ảnh gốc
        if lines is None or len(lines) == 0:
            return image
        
        # Tính toán góc nghiêng trung bình
        angles = []
        for line in lines:
            rho, theta = line[0]
            # Chỉ xét các đường ngang (góc gần 0 hoặc pi)
            if (theta < np.pi/4 or theta > 3*np.pi/4):
                angle = theta - np.pi/2
                angles.append(angle)
        
        # Nếu không có góc hợp lệ, trả về ảnh gốc
        if not angles:
            return image
        
        # Tính góc nghiêng trung bình
        median_angle = np.median(angles)
        
        # Xoay ảnh để điều chỉnh góc nghiêng
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, np.degrees(median_angle), 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Khử nhiễu từ ảnh
        
        Args:
            image: Ảnh đầu vào dạng numpy array
            
        Returns:
            np.ndarray: Ảnh đã khử nhiễu
        """
        # Sử dụng bộ lọc trung bình để khử nhiễu
        return cv2.medianBlur(image, 3)
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Áp dụng toàn bộ quy trình tiền xử lý
        
        Args:
            image: Ảnh đầu vào dạng numpy array
            
        Returns:
            torch.Tensor: Ảnh đã tiền xử lý dạng tensor
        """
        # Đảm bảo ảnh là grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Điều chỉnh góc nghiêng
        image = self.correct_skew(image)
        
        # Khử nhiễu
        image = self.denoise(image)
        
        # Áp dụng CLAHE
        image = self.apply_clahe(image)
        
        # Thay đổi kích thước
        image = self.resize_image(image)
        
        # Áp dụng ngưỡng thích ứng
        image = self.apply_adaptive_threshold(image)
        
        # Chuyển thành tensor và chuẩn hóa
        tensor = self.to_tensor(image)
        tensor = self.normalize(tensor)
        
        return tensor