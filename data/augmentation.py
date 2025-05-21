"""
augmentation.py - Các phương pháp tăng cường dữ liệu
"""
import cv2
import numpy as np
import random
from typing import Union, Tuple, List, Optional
from config.config import get_config


class Augmentation:
    """
    Lớp tăng cường dữ liệu cho ảnh biển số xe
    """
    
    def __init__(self):
        """
        Khởi tạo bộ tăng cường dữ liệu
        """
        self.config = get_config()
        
        # Lấy các tham số tăng cường từ cấu hình
        self.rotation_range = self.config.get('augmentation.rotation_range', 5)
        self.brightness_range = self.config.get('augmentation.brightness_range', (0.8, 1.2))
        self.contrast_range = self.config.get('augmentation.contrast_range', (0.8, 1.2))
        self.noise_probability = self.config.get('augmentation.noise_probability', 0.2)
        self.blur_probability = self.config.get('augmentation.blur_probability', 0.1)
        self.shadow_probability = self.config.get('augmentation.shadow_probability', 0.1)
    
    def random_rotation(self, image: np.ndarray) -> np.ndarray:
        """
        Xoay ảnh ngẫu nhiên trong khoảng rotation_range
        
        Args:
            image: Ảnh đầu vào dạng numpy array
            
        Returns:
            np.ndarray: Ảnh đã xoay
        """
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (width, height), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated_image
    
    def random_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Thay đổi độ sáng và độ tương phản ngẫu nhiên
        
        Args:
            image: Ảnh đầu vào dạng numpy array
            
        Returns:
            np.ndarray: Ảnh đã thay đổi độ sáng và độ tương phản
        """
        # Thay đổi độ sáng
        brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
        contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
        
        # Áp dụng thay đổi
        image = image.astype(np.float32)
        image = image * contrast + 10.0 * (brightness - 1.0)
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Thêm nhiễu vào ảnh
        
        Args:
            image: Ảnh đầu vào dạng numpy array
            
        Returns:
            np.ndarray: Ảnh đã thêm nhiễu
        """
        if random.random() < self.noise_probability:
            # Tạo ma trận nhiễu Gaussian
            noise = np.random.normal(0, 10, image.shape).astype(np.float32)
            
            # Thêm nhiễu vào ảnh
            noisy_image = image.astype(np.float32) + noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            
            return noisy_image
        
        return image
    
    def add_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Thêm hiệu ứng mờ vào ảnh
        
        Args:
            image: Ảnh đầu vào dạng numpy array
            
        Returns:
            np.ndarray: Ảnh đã thêm hiệu ứng mờ
        """
        if random.random() < self.blur_probability:
            # Chọn ngẫu nhiên kích thước kernel
            kernel_size = random.choice([3, 5])
            
            # Áp dụng bộ lọc Gaussian để tạo hiệu ứng mờ
            blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
            return blurred_image
        
        return image
    
    def add_shadow(self, image: np.ndarray) -> np.ndarray:
        """
        Thêm bóng đổ giả vào ảnh
        
        Args:
            image: Ảnh đầu vào dạng numpy array
            
        Returns:
            np.ndarray: Ảnh đã thêm bóng đổ
        """
        if random.random() < self.shadow_probability:
            height, width = image.shape[:2]
            
            # Tạo mặt nạ bóng đổ
            shadow_mask = np.ones_like(image)
            
            # Chọn hai điểm ngẫu nhiên để tạo đường thẳng
            x1, y1 = random.randint(0, width), 0
            x2, y2 = random.randint(0, width), height
            
            # Tạo mặt nạ tam giác
            triangle_pts = np.array([[x1, y1], [x2, y2], [0, height]], dtype=np.int32)
            cv2.fillConvexPoly(shadow_mask, triangle_pts, (0.7, 0.7, 0.7) if len(image.shape) == 3 else 0.7)
            
            # Áp dụng bóng đổ
            shadowed_image = (image * shadow_mask).astype(np.uint8)
            
            return shadowed_image
        
        return image
    
    def perspective_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Áp dụng biến đổi phối cảnh để mô phỏng góc nhìn khác
        
        Args:
            image: Ảnh đầu vào dạng numpy array
            
        Returns:
            np.ndarray: Ảnh đã áp dụng biến đổi phối cảnh
        """
        height, width = image.shape[:2]
        
        # Định nghĩa mức độ biến dạng (nhỏ để giữ tính nhận dạng được)
        distortion = width * 0.05
        
        # Tạo các điểm nguồn (hình chữ nhật gốc)
        src_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Tạo các điểm đích (biến dạng nhẹ)
        dst_points = np.array([
            [random.uniform(0, distortion), random.uniform(0, distortion)],
            [random.uniform(width - distortion, width - 1), random.uniform(0, distortion)],
            [random.uniform(width - distortion, width - 1), random.uniform(height - distortion, height - 1)],
            [random.uniform(0, distortion), random.uniform(height - distortion, height - 1)]
        ], dtype=np.float32)
        
        # Tính toán ma trận biến đổi phối cảnh
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Áp dụng biến đổi
        transformed_image = cv2.warpPerspective(
            image,
            perspective_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return transformed_image
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Áp dụng ngẫu nhiên các phương pháp tăng cường dữ liệu
        
        Args:
            image: Ảnh đầu vào dạng numpy array
            
        Returns:
            np.ndarray: Ảnh đã tăng cường
        """
        # Áp dụng các phép biến đổi với xác suất khác nhau
        
        # Xoay ảnh
        if random.random() < 0.5:
            image = self.random_rotation(image)
        
        # Thay đổi độ sáng và độ tương phản
        if random.random() < 0.7:
            image = self.random_brightness_contrast(image)
        
        # Thêm nhiễu
        image = self.add_noise(image)
        
        # Thêm hiệu ứng mờ
        image = self.add_blur(image)
        
        # Thêm bóng đổ
        image = self.add_shadow(image)
        
        # Áp dụng biến đổi phối cảnh (ít khi áp dụng vì có thể làm méo ký tự quá mức)
        if random.random() < 0.2:
            image = self.perspective_transform(image)
        
        return image