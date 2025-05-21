"""
config.py - Cấu hình cho mô hình và quá trình huấn luyện
"""
import os
import yaml
from typing import Dict, Any, Optional, List, Union
import torch


class Config:
    """
    Lớp quản lý cấu hình cho toàn bộ dự án
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Khởi tạo cấu hình từ file YAML hoặc với giá trị mặc định
        
        Args:
            config_path: Đường dẫn đến file cấu hình YAML
        """
        # Cấu hình mặc định
        self.config = {
            # Cấu hình dữ liệu
            'data': {
                'train_csv': 'dataset/labels.csv',
                'img_dir': 'dataset/images/',
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1,
                'batch_size': 32,
                'num_workers': 4,
                'image_size': (32, 140),  # (height, width)
                'max_length': 10,  # Độ dài tối đa chuỗi ký tự
                'chars': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',  # Các ký tự cần nhận dạng
            },
            
            # Cấu hình tiền xử lý và tăng cường dữ liệu
            'preprocessing': {
                'normalize_mean': [0.5],
                'normalize_std': [0.5],
                'use_clahe': True,
                'clip_limit': 2.0,
                'tile_grid_size': (8, 8),
                'adaptive_threshold': True,
            },
            
            'augmentation': {
                'enabled': True,
                'rotation_range': 5,  # Góc xoay tối đa (độ)
                'brightness_range': (0.8, 1.2),
                'contrast_range': (0.8, 1.2),
                'noise_probability': 0.2,
                'blur_probability': 0.1,
                'shadow_probability': 0.1,
            },
            
            # Cấu hình mô hình
            'model': {
                'backbone': 'efficientnet_b0',  # vgg19, resnet18, mobilenet_v3_small, efficientnet_b0
                'backbone_pretrained': True,
                'use_stn': True,
                'rnn_hidden_size': 256,
                'rnn_num_layers': 2,
                'rnn_bidirectional': True,
                'rnn_dropout': 0.2,
                'rnn_type': 'gru',  # gru, lstm
                'attention_type': 'multi_head',  # additive, multi_head
                'attention_heads': 8,
                'decoder_type': 'rnn',  # rnn, transformer
                'embedding_dim': 256,
                'transformer_nhead': 8,
                'transformer_dim_feedforward': 2048,
                'transformer_dropout': 0.1,
                'use_language_model': True,
            },
            
            # Cấu hình huấn luyện
            'training': {
                'epochs': 100,
                'optimizer': 'adam',  # adam, sgd, adamw
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'lr_scheduler': 'cosine',  # step, cosine, reduce_on_plateau
                'lr_step_size': 20,
                'lr_gamma': 0.1,
                'early_stopping_patience': 10,
                'teacher_forcing_ratio': 0.5,
                'clip_gradient': 5.0,
                'mixed_precision': True,
            },
            
            # Cấu hình loss
            'loss': {
                'type': 'combined',  # cross_entropy, ctc, focal, combined
                'label_smoothing': 0.1,
                'focal_alpha': 0.25,
                'focal_gamma': 2.0,
                'cross_entropy_weight': 1.0,
                'ctc_weight': 0.5,
                'hard_mining': True,
                'hard_mining_ratio': 0.2,
            },
            
            # Cấu hình hậu xử lý
            'postprocessing': {
                'use_beam_search': True,
                'beam_width': 5,
                'use_lexicon': True,
                'apply_license_plate_rules': True,
                'province_codes': "11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,40,41,42,43,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,88,89,90,92,93,94,95,97,98,99",
                'serie_codes': "A,B,C,D,E,F,G,H,K,L,M,N,P,S,T,U,V,X,Y,Z",
            },
            
            # Cấu hình đánh giá
            'evaluation': {
                'metrics': ['accuracy', 'edit_distance', 'character_accuracy'],
                'confusion_matrix': True,
                'log_incorrect_predictions': True,
            },
            
            # Cấu hình lưu trữ
            'checkpoint': {
                'save_dir': 'experiments/',
                'save_best_only': True,
                'save_frequency': 5,
            },
        }
        
        # Nạp cấu hình từ file YAML nếu có
        if config_path is not None and os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                self._update_config(self.config, yaml_config)
        
        # Tạo một số trường bổ sung dựa trên cấu hình
        self._post_process_config()
    
    def _update_config(self, base_config: Dict, new_config: Dict) -> None:
        """
        Cập nhật config đệ quy từ new_config vào base_config
        """
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _post_process_config(self) -> None:
        """
        Tạo các trường bổ sung sau khi nạp cấu hình
        """
        data_config = self.config['data']
        # Tạo mapping ký tự - index
        chars = data_config['chars']
        self.config['data']['char_to_idx'] = {char: idx for idx, char in enumerate(chars)}
        self.config['data']['idx_to_char'] = {idx: char for idx, char in enumerate(chars)}
        self.config['data']['num_classes'] = len(chars)
        
        # Tạo mapping mã tỉnh và serie
        if self.config['postprocessing']['apply_license_plate_rules']:
            province_codes = self.config['postprocessing']['province_codes'].split(',')
            self.config['postprocessing']['province_codes_set'] = set(province_codes)
            
            serie_codes = self.config['postprocessing']['serie_codes'].split(',')
            self.config['postprocessing']['serie_codes_set'] = set(serie_codes)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Lấy giá trị cấu hình theo key, hỗ trợ nested key với dấu chấm
        
        Args:
            key: Key cần lấy giá trị, ví dụ 'model.backbone'
            default: Giá trị mặc định nếu không tìm thấy key
            
        Returns:
            Giá trị cấu hình
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def save(self, path: str) -> None:
        """
        Lưu cấu hình hiện tại vào file YAML
        
        Args:
            path: Đường dẫn đến file cấu hình YAML
        """
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def update(self, config_dict: Dict) -> None:
        """
        Cập nhật cấu hình từ một dict mới
        
        Args:
            config_dict: Dict cấu hình mới
        """
        self._update_config(self.config, config_dict)
        self._post_process_config()
    
    def __str__(self) -> str:
        """
        Trả về chuỗi biểu diễn cấu hình
        """
        return yaml.dump(self.config, default_flow_style=False)

    def get_device(self) -> torch.device:
        """
        Lấy thiết bị dùng cho huấn luyện mô hình
        
        Returns:
            torch.device: CPU hoặc CUDA
        """
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')


# Singleton instance
CONFIG = Config()

def load_config(config_path: str) -> Config:
    """
    Tải cấu hình từ file và cập nhật vào instance CONFIG
    
    Args:
        config_path: Đường dẫn đến file cấu hình YAML
        
    Returns:
        Config: Instance cấu hình đã cập nhật
    """
    global CONFIG
    CONFIG = Config(config_path)
    return CONFIG

def get_config() -> Config:
    """
    Lấy instance CONFIG hiện tại
    
    Returns:
        Config: Instance cấu hình hiện tại
    """
    return CONFIG