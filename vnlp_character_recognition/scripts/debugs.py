"""
debug.py - Script debug để kiểm tra cấu hình và dataset
"""
import os
import sys
import torch
import pandas as pd

# Thêm thư mục gốc vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import get_config, load_config
from data.dataset import LicensePlateDataset
from data.preprocessor import Preprocessor


def debug_config():
    """
    Kiểm tra cấu hình
    """
    print("=== DEBUG CONFIG ===")
    config = get_config()
    
    print(f"Number of classes: {config.get('data.num_classes')}")
    print(f"Characters: {config.get('data.chars')}")
    print(f"char_to_idx length: {len(config.get('data.char_to_idx', {}))}")
    print(f"idx_to_char length: {len(config.get('data.idx_to_char', {}))}")
    
    # Kiểm tra special tokens
    char_to_idx = config.get('data.char_to_idx', {})
    print(f"Special tokens in char_to_idx:")
    print(f"  <pad>: {char_to_idx.get('<pad>', 'NOT FOUND')}")
    print(f"  <sos>: {char_to_idx.get('<sos>', 'NOT FOUND')}")
    print(f"  <eos>: {char_to_idx.get('<eos>', 'NOT FOUND')}")
    
    # Kiểm tra index cao nhất
    if char_to_idx:
        max_idx = max(char_to_idx.values())
        print(f"Max index in char_to_idx: {max_idx}")
        print(f"Expected num_classes: {max_idx + 1}")
    
    print()


def debug_dataset():
    """
    Kiểm tra dataset
    """
    print("=== DEBUG DATASET ===")
    config = get_config()
    
    # Kiểm tra file CSV
    csv_file = config.get('data.train_csv')
    img_dir = config.get('data.img_dir')
    
    print(f"CSV file: {csv_file}")
    print(f"Image directory: {img_dir}")
    print(f"CSV exists: {os.path.exists(csv_file)}")
    print(f"Image directory exists: {os.path.exists(img_dir)}")
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        print(f"CSV shape: {df.shape}")
        print(f"CSV columns: {list(df.columns)}")
        print(f"Sample labels: {df['license_plate'].head().tolist()}")
        
        # Kiểm tra các ký tự trong labels
        all_chars = set()
        for label in df['license_plate']:
            for char in str(label):
                all_chars.add(char)
        
        print(f"All characters in labels: {sorted(all_chars)}")
        
        # Kiểm tra ký tự không có trong char_to_idx
        char_to_idx = config.get('data.char_to_idx', {})
        missing_chars = all_chars - set(char_to_idx.keys())
        if missing_chars:
            print(f"WARNING: Characters in labels but not in char_to_idx: {sorted(missing_chars)}")
        else:
            print("All characters in labels are present in char_to_idx")
    
    print()


def debug_sample():
    """
    Kiểm tra một mẫu từ dataset
    """
    print("=== DEBUG SAMPLE ===")
    config = get_config()
    
    try:
        # Tạo dataset
        csv_file = config.get('data.train_csv')
        img_dir = config.get('data.img_dir')
        
        dataset = LicensePlateDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            transform=Preprocessor(),
            mode='train'
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # Lấy mẫu đầu tiên
            image, target, label = dataset[0]
            
            print(f"Image shape: {image.shape}")
            print(f"Image dtype: {image.dtype}")
            print(f"Target shape: {target.shape}")
            print(f"Target dtype: {target.dtype}")
            print(f"Target values: {target.tolist()}")
            print(f"Target max: {target.max().item()}")
            print(f"Target min: {target.min().item()}")
            print(f"Original label: {label}")
            
            # Decode target để kiểm tra
            idx_to_char = config.get('data.idx_to_char', {})
            decoded = []
            for idx in target.tolist():
                if idx in idx_to_char:
                    decoded.append(idx_to_char[idx])
                else:
                    decoded.append(f"[{idx}]")
            print(f"Decoded target: {''.join(decoded)}")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def main():
    """
    Main debug function
    """
    print("Starting debug session...\n")
    
    debug_config()
    debug_dataset()
    debug_sample()
    
    print("Debug session completed!")


if __name__ == '__main__':
    main()