"""
test_single_image.py - Test model VOCR với đầu vào là 1 ảnh
"""
import os
import sys
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Thêm đường dẫn root vào sys.path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from config.config import get_config, load_config
from models.vocr import VOCR
from data.preprocessor import Preprocessor
from utils.visualize import visualize_predictions


def load_trained_model(checkpoint_path: str, config):
    """
    Load model đã được train
    
    Args:
        checkpoint_path: Đường dẫn checkpoint
        config: Configuration
        
    Returns:
        model: Model đã load weights
    """
    try:
        # Tạo model
        model = VOCR()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Model loaded from: {checkpoint_path}")
        print(f"📊 Model info: {model.get_model_info()}")
        
        if 'epoch' in checkpoint:
            print(f"🔄 Epoch: {checkpoint['epoch'] + 1}")
        if 'best_val_loss' in checkpoint:
            print(f"📉 Best validation loss: {checkpoint['best_val_loss']:.4f}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def preprocess_single_image(image_path: str, target_size=(32, 140)):
    """
    Tiền xử lý 1 ảnh đầu vào
    
    Args:
        image_path: Đường dẫn ảnh
        target_size: Kích thước target (height, width)
        
    Returns:
        torch.Tensor: Ảnh đã được tiền xử lý
    """
    try:
        # Đọc ảnh
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Đọc ảnh bằng OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        print(f"📷 Original image shape: {image.shape}")
        
        # Resize ảnh
        image_resized = cv2.resize(image, (target_size[1], target_size[0]))
        print(f"🔄 Resized to: {image_resized.shape}")
        
        # Sử dụng preprocessor
        preprocessor = Preprocessor()
        image_tensor = preprocessor(image_resized)
        
        # Thêm batch dimension
        image_tensor = image_tensor.unsqueeze(0)  # [1, 1, height, width]
        
        print(f"✅ Preprocessed tensor shape: {image_tensor.shape}")
        
        return image_tensor, image_resized
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None, None


def format_license_plate(text: str) -> str:
    """
    Format license plate với dấu gạch ngang
    
    Args:
        text: License plate text
        
    Returns:
        str: Formatted text với dấu gạch ngang
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Làm sạch text
    text = text.replace(' ', '').upper().strip()
    
    # Nếu đã có dấu gạch ngang, giữ nguyên
    if '-' in text:
        return text
    
    # Thêm dấu gạch ngang theo format Việt Nam: 12A-34567
    # Tìm vị trí cuối cùng của chữ cái
    last_letter_pos = -1
    for i, char in enumerate(text):
        if char.isalpha():
            last_letter_pos = i
    
    # Nếu tìm thấy chữ cái và có số sau đó, thêm dấu gạch ngang
    if last_letter_pos >= 0 and last_letter_pos < len(text) - 1:
        formatted = text[:last_letter_pos + 1] + '-' + text[last_letter_pos + 1:]
        return formatted
    
    return text


def predict_single_image(model, image_tensor, config, device='cpu'):
    """
    Dự đoán biển số xe từ 1 ảnh
    
    Args:
        model: Model VOCR
        image_tensor: Tensor ảnh đã tiền xử lý
        config: Configuration
        device: Device để chạy
        
    Returns:
        str: Biển số xe được dự đoán
    """
    try:
        # Chuyển model và tensor lên device
        model = model.to(device)
        image_tensor = image_tensor.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = model(image_tensor, targets=None, teacher_forcing_ratio=0.0)
            logits = outputs['outputs']  # [1, seq_len, num_classes]
            
            # Get predicted indices
            _, predicted_indices = torch.max(logits, dim=2)  # [1, seq_len]
            
            # Decode to string
            indices = predicted_indices[0].cpu().tolist()
            chars = []
            
            idx_to_char = config.get('data.idx_to_char', {})
            
            for idx in indices:
                if idx == 0:  # <pad>
                    continue
                elif idx == 1:  # <sos>
                    continue
                elif idx == 2:  # <eos>
                    break
                elif idx in idx_to_char:
                    chars.append(idx_to_char[idx])
            
            prediction = ''.join(chars)
            
            # Format với dấu gạch ngang
            formatted_prediction = format_license_plate(prediction)
            
            print(f"🔤 Raw prediction: '{prediction}'")
            print(f"📋 Formatted prediction: '{formatted_prediction}'")
            
            return formatted_prediction, outputs.get('attention_weights')
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return "", None


def visualize_result(original_image, prediction, attention_weights=None, save_path=None):
    """
    Hiển thị kết quả dự đoán
    
    Args:
        original_image: Ảnh gốc
        prediction: Kết quả dự đoán
        attention_weights: Attention weights (optional)
        save_path: Đường dẫn lưu hình (optional)
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Hiển thị ảnh gốc
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title(f'Input Image', fontsize=14)
        axes[0].axis('off')
        
        # Hiển thị kết quả
        axes[1].text(0.5, 0.7, 'Prediction:', ha='center', va='center', 
                    fontsize=16, fontweight='bold', transform=axes[1].transAxes)
        axes[1].text(0.5, 0.4, f"'{prediction}'", ha='center', va='center', 
                    fontsize=20, color='red', fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    transform=axes[1].transAxes)
        
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Result saved to: {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error visualizing result: {e}")


def test_multiple_images(model, image_folder, config, device='cpu', max_images=10):
    """
    Test model với nhiều ảnh trong folder
    
    Args:
        model: Model VOCR
        image_folder: Đường dẫn folder chứa ảnh
        config: Configuration
        device: Device
        max_images: Số lượng ảnh tối đa để test
    """
    try:
        # Lấy danh sách file ảnh
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f'*{ext}'))
            image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
        
        image_files = sorted(image_files)[:max_images]
        
        if not image_files:
            print(f"❌ No images found in {image_folder}")
            return
        
        print(f"🔍 Found {len(image_files)} images to test")
        
        results = []
        
        for i, image_path in enumerate(image_files):
            print(f"\n{'='*50}")
            print(f"📷 Testing image {i+1}/{len(image_files)}: {image_path.name}")
            print('='*50)
            
            # Preprocess image
            image_tensor, original_image = preprocess_single_image(str(image_path))
            
            if image_tensor is not None:
                # Predict
                prediction, attention_weights = predict_single_image(
                    model, image_tensor, config, device
                )
                
                results.append({
                    'image_path': str(image_path),
                    'prediction': prediction
                })
                
                print(f"✅ Result: {image_path.name} -> '{prediction}'")
            else:
                print(f"❌ Failed to process: {image_path.name}")
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 SUMMARY RESULTS")
        print('='*60)
        for i, result in enumerate(results):
            print(f"{i+1:2d}. {Path(result['image_path']).name:<30} -> '{result['prediction']}'")
        
        return results
        
    except Exception as e:
        print(f"❌ Error testing multiple images: {e}")
        return []


def main():
    """
    Hàm main
    """
    parser = argparse.ArgumentParser(description='Test VOCR Model with Single Image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image or folder')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save result image')
    parser.add_argument('--batch', action='store_true',
                       help='Test multiple images in folder')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config()
    
    print("🚀 VOCR Model Testing")
    print("="*50)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"💻 Using device: {device}")
    
    # Load model
    print(f"📂 Loading model from: {args.checkpoint}")
    model = load_trained_model(args.checkpoint, config)
    
    if model is None:
        print("❌ Failed to load model. Exiting...")
        return
    
    model = model.to(device)
    
    # Test mode: single image or batch
    if args.batch or os.path.isdir(args.image):
        print(f"📁 Batch testing mode: {args.image}")
        results = test_multiple_images(model, args.image, config, device)
        
    else:
        print(f"🖼️  Single image testing: {args.image}")
        
        # Preprocess image
        print("\n🔄 Preprocessing image...")
        image_tensor, original_image = preprocess_single_image(args.image)
        
        if image_tensor is None:
            print("❌ Failed to preprocess image. Exiting...")
            return
        
        # Predict
        print("\n🧠 Running prediction...")
        prediction, attention_weights = predict_single_image(
            model, image_tensor, config, device
        )
        
        # Display results
        print(f"\n🎯 FINAL RESULT: '{prediction}'")
        
        # Visualize
        print("\n📊 Visualizing result...")
        save_path = args.save if args.save else None
        visualize_result(original_image, prediction, attention_weights, save_path)
        
        # Additional info
        print(f"\n📋 Prediction details:")
        print(f"   - Length: {len(prediction)} characters")
        print(f"   - Format: Vietnamese license plate")
        print(f"   - Confidence: Model prediction (no confidence score available)")
    
    print("\n✅ Testing completed!")


if __name__ == '__main__':
    main()