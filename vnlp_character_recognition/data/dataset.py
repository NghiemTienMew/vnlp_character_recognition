"""
dataset.py - Module xử lý dữ liệu và tạo DataLoader (đã sửa lỗi)
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from typing import Tuple, List, Dict, Optional, Union, Callable
import random
from pathlib import Path

from config.config import get_config
from data.preprocessor import Preprocessor
from data.augmentation import Augmentation


class LicensePlateDataset(Dataset):
    """
    Dataset cho dữ liệu biển số xe
    """
    
    def __init__(
        self, 
        csv_file: str, 
        img_dir: str, 
        transform: Optional[Callable] = None, 
        augment: Optional[Callable] = None,
        mode: str = 'train'
    ):
        """
        Khởi tạo dataset
        
        Args:
            csv_file: Đường dẫn đến file CSV chứa thông tin ảnh và nhãn
            img_dir: Thư mục chứa ảnh
            transform: Các phép biển đổi tiền xử lý
            augment: Các phép tăng cường dữ liệu
            mode: 'train', 'val', hoặc 'test'
        """
        self.config = get_config()
        self.img_dir = img_dir
        self.mode = mode
        
        # Đọc file CSV
        self.data_df = pd.read_csv(csv_file)
        
        # Xử lý đường dẫn ảnh
        if not os.path.isabs(img_dir):
            # Nếu img_dir là đường dẫn tương đối, thêm vào đường dẫn của mỗi ảnh
            self.data_df['image_path'] = self.data_df['image_path'].apply(
                lambda x: os.path.join(img_dir, os.path.basename(x)) if not x.startswith(img_dir) else x
            )
        
        # Chia dữ liệu theo chế độ
        if mode != 'predict':
            self._split_dataset()
        
        # Chuyển đổi nhãn thành indices
        self.char_to_idx = self.config.get('data.char_to_idx')
        self.idx_to_char = self.config.get('data.idx_to_char')
        
        # In ra một số thông tin để debug
        print(f"Dataset mode: {mode}, Size: {len(self.data_df)}")
        print(f"Number of classes: {self.config.get('data.num_classes')}")
        
        # Thiết lập transformations
        self.transform = transform
        self.augment = augment
        if self.transform is None:
            self.transform = Preprocessor()
        if self.augment is None and mode == 'train' and self.config.get('augmentation.enabled', True):
            self.augment = Augmentation()
        
        # Độ dài tối đa của chuỗi ký tự
        self.max_length = self.config.get('data.max_length', 10)
    
    def __len__(self) -> int:
        """
        Trả về số lượng mẫu trong dataset
        """
        return len(self.data_df)
    
    def _split_dataset(self):
        """
        Chia dữ liệu thành train, val, test
        """
        train_ratio = self.config.get('data.train_ratio', 0.8)
        val_ratio = self.config.get('data.val_ratio', 0.1)
        test_ratio = self.config.get('data.test_ratio', 0.1)
        
        # Đảm bảo tổng tỷ lệ là 1
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        
        # Tạo một cột chứa dữ liệu ngẫu nhiên để chia
        np.random.seed(42)
        self.data_df['random'] = np.random.random(len(self.data_df))
        
        # Chia dữ liệu
        if self.mode == 'train':
            self.data_df = self.data_df[self.data_df['random'] < train_ratio]
        elif self.mode == 'val':
            self.data_df = self.data_df[(self.data_df['random'] >= train_ratio) & 
                                        (self.data_df['random'] < train_ratio + val_ratio)]
        elif self.mode == 'test':
            self.data_df = self.data_df[self.data_df['random'] >= train_ratio + val_ratio]
        
        # Xóa cột random
        self.data_df.drop('random', axis=1, inplace=True)
    
    def _preprocess_one_row_two_rows(self, img_path: str) -> np.ndarray:
        """
        Xử lý ảnh biển số 1 dòng hoặc 2 dòng
        
        Args:
            img_path: Đường dẫn đến ảnh
            
        Returns:
            np.ndarray: Ảnh đã xử lý
        """
        # Đọc ảnh
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Unable to read image at {img_path}")
            # Tạo một ảnh trống nếu không đọc được
            image = np.zeros((32, 140), dtype=np.uint8)
            return image
        
        # Nếu ảnh 2 dòng, chia thành 2 dòng và lấy dòng đầu tiên
        filename = os.path.basename(img_path)
        if filename.startswith('two_rows'):
            # Sử dụng thuật toán X-Y Cut để chia ảnh thành 2 dòng
            h, w = image.shape
            # Tính tổng giá trị pixel theo trục y
            y_projection = np.sum(image, axis=1)
            
            # Tìm vị trí giữa hai dòng (thường là vị trí có giá trị tổng thấp)
            # Tìm trong khoảng 1/3 đến 2/3 chiều cao
            search_start = h // 3
            search_end = h * 2 // 3
            search_region = y_projection[search_start:search_end]
            
            # Kiểm tra để tránh lỗi khi search_region trống
            if len(search_region) > 0:
                # Tìm vị trí có giá trị thấp nhất trong vùng tìm kiếm
                min_idx = np.argmin(search_region) + search_start
                
                # Chia ảnh thành 2 phần
                top_plate = image[:min_idx, :]
                bottom_plate = image[min_idx:, :]
                
                # Dùng dòng đầu (dòng trên)
                image = top_plate
            
        return image
    
    def _encode_label(self, label: str) -> torch.Tensor:
        """
        Chuyển đổi chuỗi ký tự thành tensor indices
        
        Args:
            label: Chuỗi ký tự nhãn
            
        Returns:
            torch.Tensor: Tensor chứa chỉ số của các ký tự
        """
        # Thêm dấu gạch ngang nếu cần thiết
        if '-' not in label:
            # Biển số xe Việt Nam thường có dạng: 2 số đầu là mã tỉnh, sau đó là serie (1-2 ký tự), sau đó là 5 số
            # Thêm dấu gạch ngang sau 2 số đầu và serie
            if len(label) >= 4:  # Đảm bảo đủ dài để xử lý
                # Duyệt từ ký tự thứ 3 để tìm ký tự chữ cái cuối cùng
                last_letter_idx = 2
                for i in range(2, min(4, len(label))):
                    if label[i].isalpha():
                        last_letter_idx = i
                
                # Thêm dấu gạch ngang sau ký tự chữ cái cuối cùng
                label = label[:last_letter_idx+1] + '-' + label[last_letter_idx+1:]
        
        # Chuyển đổi thành indices
        indices = []
        for c in label:
            if c in self.char_to_idx:
                indices.append(self.char_to_idx[c])
            else:
                # Nếu ký tự không nằm trong char_to_idx, bỏ qua
                print(f"Warning: Character '{c}' not found in char_to_idx")
        
        # Thêm token bắt đầu (<sos>) và kết thúc (<eos>)
        sos_idx = self.char_to_idx.get('<sos>', 1)  # Mặc định là 1
        eos_idx = self.char_to_idx.get('<eos>', 2)  # Mặc định là 2
        
        indices = [sos_idx] + indices + [eos_idx]
        
        # Padding để đảm bảo độ dài cố định
        pad_idx = self.char_to_idx.get('<pad>', 0)  # Mặc định là 0
        while len(indices) < self.max_length + 2:  # +2 cho <sos> và <eos>
            indices.append(pad_idx)
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Lấy một mẫu từ dataset
        
        Args:
            idx: Chỉ số của mẫu
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: 
                - Ảnh đã xử lý dạng tensor
                - Nhãn dạng tensor
                - Chuỗi nhãn gốc
        """
        # Lấy thông tin từ DataFrame
        img_path = self.data_df.iloc[idx]['image_path']
        label = self.data_df.iloc[idx]['license_plate']
        
        # Đọc và xử lý ảnh
        image = self._preprocess_one_row_two_rows(img_path)
        
        # Áp dụng data augmentation nếu cần
        if self.augment is not None and self.mode == 'train':
            image = self.augment(image)
        
        # Áp dụng các phép biến đổi tiền xử lý
        if self.transform is not None:
            image = self.transform(image)
        
        # Encode nhãn
        encoded_label = self._encode_label(label)
        
        return image, encoded_label, label


def create_data_loaders():
    """
    Tạo DataLoader cho tập train, validation và test
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoader cho train, val và test
    """
    config = get_config()
    
    # Lấy đường dẫn từ cấu hình
    csv_file = config.get('data.train_csv')
    img_dir = config.get('data.img_dir')
    batch_size = config.get('data.batch_size', 32)
    num_workers = min(4, os.cpu_count() or 1)  # Giảm xuống để tránh cảnh báo
    
    # Tạo bộ tiền xử lý
    preprocessor = Preprocessor()
    
    # Tạo bộ tăng cường dữ liệu
    augmentation = Augmentation() if config.get('augmentation.enabled', True) else None
    
    # Tạo dataset
    train_dataset = LicensePlateDataset(
        csv_file=csv_file,
        img_dir=img_dir,
        transform=preprocessor,
        augment=augmentation,
        mode='train'
    )
    
    val_dataset = LicensePlateDataset(
        csv_file=csv_file,
        img_dir=img_dir,
        transform=preprocessor,
        augment=None,  # Không dùng augmentation cho validation
        mode='val'
    )
    
    test_dataset = LicensePlateDataset(
        csv_file=csv_file,
        img_dir=img_dir,
        transform=preprocessor,
        augment=None,  # Không dùng augmentation cho test
        mode='test'
    )
    
    # Tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader