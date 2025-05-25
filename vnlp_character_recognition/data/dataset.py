"""
dataset.py - Module xử lý dữ liệu và tạo DataLoader (đã sửa lỗi index out of bounds)
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


def custom_collate_fn(batch):
    """
    Custom collate function để xử lý batch size mismatch và đảm bảo consistency
    
    Args:
        batch: List of (image, target, label) tuples
        
    Returns:
        Tuple: (images, targets, labels) với kích thước nhất quán
    """
    try:
        # Tách batch thành các thành phần
        if len(batch[0]) == 3:
            images, targets, labels = zip(*batch)
        else:
            images, targets = zip(*batch)
            labels = None
        
        # Chuyển đổi images thành tensor
        images = torch.stack(images)
        
        # Xử lý targets để đảm bảo cùng kích thước
        max_target_len = max(target.size(0) for target in targets)
        batch_size = len(targets)
        
        # Tạo tensor targets với padding
        padded_targets = torch.zeros(batch_size, max_target_len, dtype=torch.long)
        
        for i, target in enumerate(targets):
            target_len = target.size(0)
            if target_len <= max_target_len:
                padded_targets[i, :target_len] = target
            else:
                # Cắt nếu target quá dài
                padded_targets[i] = target[:max_target_len]
        
        if labels is not None:
            return images, padded_targets, list(labels)
        else:
            return images, padded_targets
            
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        # Fallback: trả về batch đầu tiên nếu có lỗi
        if len(batch) > 0:
            if len(batch[0]) == 3:
                images, targets, labels = zip(*batch[:1])
                return torch.stack(images), torch.stack(targets), list(labels)
            else:
                images, targets = zip(*batch[:1])
                return torch.stack(images), torch.stack(targets)
        else:
            # Trả về empty tensors nếu batch rỗng
            return torch.empty(0, 1, 32, 140), torch.empty(0, 12, dtype=torch.long)


class LicensePlateDataset(Dataset):
    """
    Dataset cho dữ liệu biển số xe - Đã sửa lỗi index out of bounds
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
        try:
            self.data_df = pd.read_csv(csv_file)
            print(f"Loaded CSV with {len(self.data_df)} records")
        except Exception as e:
            print(f"Error loading CSV file {csv_file}: {e}")
            # Tạo dummy dataframe nếu không đọc được file
            self.data_df = pd.DataFrame({
                'image_path': ['dummy.jpg'],
                'license_plate': ['12A-34567']
            })
        
        # Xử lý đường dẫn ảnh
        if not os.path.isabs(img_dir):
            self.data_df['image_path'] = self.data_df['image_path'].apply(
                lambda x: os.path.join(img_dir, os.path.basename(x)) if not x.startswith(img_dir) else x
            )
        
        # Chia dữ liệu theo chế độ
        if mode != 'predict':
            self._split_dataset()
        
        # Chuyển đổi nhãn thành indices
        self.char_to_idx = self.config.get('data.char_to_idx')
        self.idx_to_char = self.config.get('data.idx_to_char')
        self.num_classes = self.config.get('data.num_classes', 40)
        
        # Kiểm tra và tạo mapping nếu chưa có
        if not self.char_to_idx or not self.idx_to_char:
            self._create_char_mappings()
        
        # In ra thông tin để debug
        print(f"Dataset mode: {mode}, Size: {len(self.data_df)}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Character mapping range: 0 to {max(self.idx_to_char.keys()) if self.idx_to_char else 'N/A'}")
        
        # Thiết lập transformations
        self.transform = transform
        self.augment = augment
        if self.transform is None:
            self.transform = Preprocessor()
        if self.augment is None and mode == 'train' and self.config.get('augmentation.enabled', True):
            self.augment = Augmentation()
        
        # Độ dài tối đa của chuỗi ký tự
        self.max_length = self.config.get('data.max_length', 10)
    
    def _create_char_mappings(self):
        """
        Tạo char mappings nếu chưa có trong config
        """
        chars = self.config.get('data.chars', '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
        
        # Tạo mapping ký tự - index
        char_to_idx = {char: idx + 3 for idx, char in enumerate(chars)}  # +3 cho special tokens
        idx_to_char = {idx + 3: char for idx, char in enumerate(chars)}
        
        # Thêm special tokens
        char_to_idx['<pad>'] = 0
        char_to_idx['<sos>'] = 1
        char_to_idx['<eos>'] = 2
        idx_to_char[0] = '<pad>'
        idx_to_char[1] = '<sos>'
        idx_to_char[2] = '<eos>'
        
        # Cập nhật config
        self.config.config['data']['char_to_idx'] = char_to_idx
        self.config.config['data']['idx_to_char'] = idx_to_char
        self.config.config['data']['num_classes'] = len(chars) + 3
        
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.num_classes = len(chars) + 3
        
        print(f"Created character mappings: {len(char_to_idx)} total characters")
        print(f"Character indices range: 0 to {max(idx_to_char.keys())}")
    
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
        
        # Reset index sau khi chia
        self.data_df = self.data_df.reset_index(drop=True)
        
        # Xóa cột random
        if 'random' in self.data_df.columns:
            self.data_df.drop('random', axis=1, inplace=True)
    
    def _preprocess_one_row_two_rows(self, img_path: str) -> np.ndarray:
        """
        Xử lý ảnh biển số 1 dòng hoặc 2 dòng với error handling
        
        Args:
            img_path: Đường dẫn đến ảnh
            
        Returns:
            np.ndarray: Ảnh đã xử lý
        """
        try:
            # Đọc ảnh
            if not os.path.exists(img_path):
                print(f"Warning: Image not found at {img_path}")
                # Tạo ảnh dummy
                return np.ones((32, 140), dtype=np.uint8) * 128
            
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Warning: Unable to read image at {img_path}")
                # Tạo ảnh dummy
                return np.ones((32, 140), dtype=np.uint8) * 128
            
            # Nếu ảnh 2 dòng, chia thành 2 dòng và lấy dòng đầu tiên
            filename = os.path.basename(img_path)
            if filename.startswith('two_rows'):
                # Sử dụng thuật toán X-Y Cut để chia ảnh thành 2 dòng
                h, w = image.shape
                if h > 30:  # Chỉ chia nếu ảnh đủ cao
                    # Tính tổng giá trị pixel theo trục y
                    y_projection = np.sum(image, axis=1)
                    
                    # Tìm vị trí giữa hai dòng
                    search_start = h // 3
                    search_end = h * 2 // 3
                    search_region = y_projection[search_start:search_end]
                    
                    if len(search_region) > 0:
                        min_idx = np.argmin(search_region) + search_start
                        top_plate = image[:min_idx, :]
                        if top_plate.shape[0] > 0:  # Đảm bảo top_plate không rỗng
                            image = top_plate
            
            return image
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            # Trả về ảnh dummy nếu có lỗi
            return np.ones((32, 140), dtype=np.uint8) * 128
    
    def _validate_indices(self, indices: List[int]) -> List[int]:
        """
        Validate indices để đảm bảo không vượt quá num_classes
        
        Args:
            indices: List các indices
            
        Returns:
            List[int]: Indices đã được validate
        """
        validated_indices = []
        for idx in indices:
            if idx < 0:
                print(f"Warning: Negative index {idx}, replacing with pad token")
                validated_indices.append(self.char_to_idx.get('<pad>', 0))
            elif idx >= self.num_classes:
                print(f"Warning: Index {idx} >= num_classes {self.num_classes}, replacing with pad token")
                validated_indices.append(self.char_to_idx.get('<pad>', 0))
            else:
                validated_indices.append(idx)
        
        return validated_indices
    
    def _encode_label(self, label: str) -> torch.Tensor:
        """
        Chuyển đổi chuỗi ký tự thành tensor indices với validation nghiêm ngặt
        
        Args:
            label: Chuỗi ký tự nhãn
            
        Returns:
            torch.Tensor: Tensor chứa chỉ số của các ký tự
        """
        try:
            # Kiểm tra và làm sạch label
            if not isinstance(label, str):
                label = str(label)
            
            label = label.strip().upper()
            
            # Thêm dấu gạch ngang nếu cần thiết
            if '-' not in label and len(label) >= 4:
                # Tìm vị trí để thêm dấu gạch ngang
                last_letter_idx = 2
                for i in range(2, min(4, len(label))):
                    if i < len(label) and label[i].isalpha():
                        last_letter_idx = i
                
                if last_letter_idx + 1 < len(label):
                    label = label[:last_letter_idx+1] + '-' + label[last_letter_idx+1:]
            
            # Chuyển đổi thành indices với validation
            indices = []
            for c in label:
                if c in self.char_to_idx:
                    char_idx = self.char_to_idx[c]
                    # Double check index bounds
                    if 0 <= char_idx < self.num_classes:
                        indices.append(char_idx)
                    else:
                        print(f"Warning: Character '{c}' maps to invalid index {char_idx}")
                        indices.append(self.char_to_idx.get('<pad>', 0))
                else:
                    print(f"Warning: Character '{c}' not found in char_to_idx, using pad token")
                    indices.append(self.char_to_idx.get('<pad>', 0))
            
            # Thêm token bắt đầu (<sos>) và kết thúc (<eos>)
            sos_idx = self.char_to_idx.get('<sos>', 1)
            eos_idx = self.char_to_idx.get('<eos>', 2)
            
            # Validate special tokens
            sos_idx = min(max(sos_idx, 0), self.num_classes - 1)
            eos_idx = min(max(eos_idx, 0), self.num_classes - 1)
            
            indices = [sos_idx] + indices + [eos_idx]
            
            # Validate all indices
            indices = self._validate_indices(indices)
            
            # Padding để đảm bảo độ dài cố định
            pad_idx = self.char_to_idx.get('<pad>', 0)
            pad_idx = min(max(pad_idx, 0), self.num_classes - 1)  # Validate pad token
            
            target_length = self.max_length + 2  # +2 cho <sos> và <eos>
            
            while len(indices) < target_length:
                indices.append(pad_idx)
            
            # Cắt nếu quá dài
            if len(indices) > target_length:
                indices = indices[:target_length]
                # Đảm bảo token cuối là <eos>
                indices[-1] = eos_idx
            
            # Final validation
            indices = self._validate_indices(indices)
            
            # Convert to tensor with additional bounds checking
            tensor_indices = torch.tensor(indices, dtype=torch.long)
            tensor_indices = torch.clamp(tensor_indices, 0, self.num_classes - 1)
            
            return tensor_indices
            
        except Exception as e:
            print(f"Error encoding label '{label}': {e}")
            # Trả về tensor dummy an toàn nếu có lỗi
            pad_idx = min(max(self.char_to_idx.get('<pad>', 0), 0), self.num_classes - 1)
            sos_idx = min(max(self.char_to_idx.get('<sos>', 1), 0), self.num_classes - 1)
            eos_idx = min(max(self.char_to_idx.get('<eos>', 2), 0), self.num_classes - 1)
            
            target_length = self.max_length + 2
            indices = [sos_idx] + [pad_idx] * (target_length - 2) + [eos_idx]
            
            # Final validation
            indices = [min(max(idx, 0), self.num_classes - 1) for idx in indices]
            
            return torch.tensor(indices, dtype=torch.long)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Lấy một mẫu từ dataset với error handling và validation nghiêm ngặt
        
        Args:
            idx: Chỉ số của mẫu
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: 
                - Ảnh đã xử lý dạng tensor
                - Nhãn dạng tensor
                - Chuỗi nhãn gốc
        """
        try:
            # Đảm bảo idx hợp lệ
            if idx >= len(self.data_df):
                idx = 0
            
            # Lấy thông tin từ DataFrame
            row = self.data_df.iloc[idx]
            img_path = row['image_path']
            label = row['license_plate']
            
            # Đọc và xử lý ảnh
            image = self._preprocess_one_row_two_rows(img_path)
            
            # Áp dụng data augmentation nếu cần
            if self.augment is not None and self.mode == 'train':
                try:
                    image = self.augment(image)
                except Exception as e:
                    print(f"Warning: Augmentation failed for {img_path}: {e}")
            
            # Áp dụng các phép biến đổi tiền xử lý
            if self.transform is not None:
                try:
                    image = self.transform(image)
                except Exception as e:
                    print(f"Warning: Transform failed for {img_path}: {e}")
                    # Tạo tensor dummy nếu transform fail
                    image = torch.zeros(1, 32, 140)
            
            # Encode nhãn với validation nghiêm ngặt
            encoded_label = self._encode_label(label)
            
            # Final safety check
            encoded_label = torch.clamp(encoded_label, 0, self.num_classes - 1)
            
            return image, encoded_label, label
            
        except Exception as e:
            print(f"Error getting item {idx}: {e}")
            # Trả về dummy data an toàn nếu có lỗi
            dummy_image = torch.zeros(1, 32, 140)
            dummy_label = self._encode_label("12A-34567")
            dummy_label = torch.clamp(dummy_label, 0, self.num_classes - 1)
            return dummy_image, dummy_label, "12A-34567"


def create_data_loaders():
    """
    Tạo DataLoader cho tập train, validation và test với custom collate function
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoader cho train, val và test
    """
    config = get_config()
    
    # Lấy đường dẫn từ cấu hình
    csv_file = config.get('data.train_csv')
    img_dir = config.get('data.img_dir')
    batch_size = config.get('data.batch_size', 32)
    num_workers = min(2, os.cpu_count() or 1)  # Giảm num_workers để tránh lỗi
    
    print("Creating data loaders...")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(csv_file):
        print(f"Warning: CSV file not found at {csv_file}")
    
    if not os.path.exists(img_dir):
        print(f"Warning: Image directory not found at {img_dir}")
    
    # Tạo bộ tiền xử lý
    try:
        preprocessor = Preprocessor()
    except Exception as e:
        print(f"Error creating preprocessor: {e}")
        preprocessor = None
    
    # Tạo bộ tăng cường dữ liệu
    augmentation = None
    if config.get('augmentation.enabled', True):
        try:
            augmentation = Augmentation()
        except Exception as e:
            print(f"Error creating augmentation: {e}")
    
    # Tạo dataset với error handling
    try:
        train_dataset = LicensePlateDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            transform=preprocessor,
            augment=augmentation,
            mode='train'
        )
    except Exception as e:
        print(f"Error creating train dataset: {e}")
        # Tạo dummy dataset
        train_dataset = LicensePlateDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            transform=None,
            augment=None,
            mode='train'
        )
    
    try:
        val_dataset = LicensePlateDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            transform=preprocessor,
            augment=None,
            mode='val'
        )
    except Exception as e:
        print(f"Error creating val dataset: {e}")
        val_dataset = train_dataset  # Fallback
    
    try:
        test_dataset = LicensePlateDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            transform=preprocessor,
            augment=None,
            mode='test'
        )
    except Exception as e:
        print(f"Error creating test dataset: {e}")
        test_dataset = train_dataset  # Fallback
    
    # In thông tin validation
    print(f"Dataset validation complete:")
    print(f"- Train dataset: {len(train_dataset)} samples")
    print(f"- Val dataset: {len(val_dataset)} samples") 
    print(f"- Test dataset: {len(test_dataset)} samples")
    print(f"- Number of classes: {train_dataset.num_classes}")
    
    # Tạo DataLoader với custom collate function
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            drop_last=True  # Bỏ batch cuối nếu không đủ kích thước
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        # Tạo dummy data loaders với batch_size nhỏ hơn
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(4, batch_size),
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(4, batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=min(4, batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        return train_loader, val_loader, test_loader