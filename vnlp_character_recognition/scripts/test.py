"""
test.py - Test model VOCR với đầu vào là 1 ảnh (Fixed for PyTorch 2.6)
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
root_path = Path(__file__).parent
sys.path.append(str(root_path))

from config.config import get_config, load_config
from models.vocr import VOCR
from data.preprocessor import Preprocessor


def load_trained_model(checkpoint_path: str, config):
    """
    Load model đã được train (Fixed for PyTorch 2.6 with safe loading)
    
    Args:
        checkpoint_path: Đường dẫn checkpoint
        config: Configuration
        
    Returns:
        model: Model đã load weights
    """
    try:
        # Tạo model
        model = VOCR()
        
        print("🔧 Method 1: Loading with safe globals...")
        # SỬA LỖI: Sử dụng safe_globals để load scheduler và optimizer
        with torch.serialization.safe_globals([torch.optim.lr_scheduler.CosineAnnealingLR]):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            
        # Nếu checkpoint chỉ chứa model weights
        if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
            model.load_state_dict(checkpoint)
            print(f"✅ Model state dict loaded directly")
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model state dict loaded from checkpoint")
            
        print(f"📊 Model info: {model.get_model_info()}")
        return model
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        print("🔧 Method 2: Loading full checkpoint with weights_only=False...")
        
        try:
            # Method 2: Load with weights_only=False (less safe but works)
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            model = VOCR()
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Model loaded from checkpoint")
                
                if 'epoch' in checkpoint:
                    print(f"🔄 Epoch: {checkpoint['epoch'] + 1}")
                if 'best_val_loss' in checkpoint or 'val_loss' in checkpoint:
                    loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', 'N/A'))
                    print(f"📉 Validation loss: {loss}")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ Model state dict loaded directly")
            
            print(f"📊 Model info: {model.get_model_info()}")
            return model
            
        except Exception as e2:
            print(f"❌ Method 2 failed: {e2}")
            print("🔧 Method 3: Manual extraction of model weights...")
            
            try:
                # Method 3: Manually extract only model weights
                import pickle
                
                # Load raw pickle data
                with open(checkpoint_path, 'rb') as f:
                    data = pickle.load(f)
                
                model = VOCR()
                
                # Try to find model state dict in the data
                if isinstance(data, dict):
                    if 'model_state_dict' in data:
                        model.load_state_dict(data['model_state_dict'])
                        print(f"✅ Model weights extracted manually")
                    elif any('weight' in key or 'bias' in key for key in data.keys()):
                        # Data itself is the state dict
                        model.load_state_dict(data)
                        print(f"✅ Model state dict loaded manually")
                    else:
                        raise ValueError("Cannot find model weights in checkpoint")
                else:
                    raise ValueError("Checkpoint format not supported")
                
                print(f"📊 Model info: {model.get_model_info()}")
                return model
                
            except Exception as e3:
                print(f"❌ Method 3 failed: {e3}")
                print("🔧 Method 4: Creating fresh model (for testing purposes)...")
                
                try:
                    # Method 4: Create fresh model with random weights (last resort)
                    model = VOCR()
                    print("⚠️  WARNING: Using fresh model with random weights!")
                    print("⚠️  This is only for testing the inference pipeline.")
                    print("⚠️  Results will be meaningless!")
                    return model
                    
                except Exception as e4:
                    print(f"❌ All methods failed: {e4}")
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
        
        # Manual preprocessing (thay vì dùng Preprocessor class)
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Normalize to [-1, 1] (như trong Preprocessor)
        image_normalized = (image_normalized - 0.5) / 0.5
        
        # Convert to tensor
        image_tensor = torch.FloatTensor(image_normalized).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        print(f"✅ Preprocessed tensor shape: {image_tensor.shape}")
        print(f"📊 Tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        
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
    
    # Thêm dấu gạch ngang theo format Việt Nam: 36A-02391
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
        
        print(f"🧠 Running inference on {device}...")
        
        with torch.no_grad():
            # Forward pass
            outputs = model(image_tensor, targets=None, teacher_forcing_ratio=0.0)
            logits = outputs['outputs']  # [1, seq_len, num_classes]
            
            print(f"📊 Model output shape: {logits.shape}")
            
            # Get predicted indices
            _, predicted_indices = torch.max(logits, dim=2)  # [1, seq_len]
            
            # Decode to string
            indices = predicted_indices[0].cpu().tolist()
            chars = []
            
            idx_to_char = config.get('data.idx_to_char', {})
            print(f"📖 Available characters: {len(idx_to_char)} chars")
            
            print(f"🔢 Predicted indices: {indices}")
            
            for idx in indices:
                if idx == 0:  # <pad>
                    continue
                elif idx == 1:  # <sos>
                    continue
                elif idx == 2:  # <eos>
                    break
                elif idx in idx_to_char:
                    chars.append(idx_to_char[idx])
                    print(f"  {idx} -> '{idx_to_char[idx]}'")
                else:
                    print(f"  {idx} -> UNKNOWN")
            
            prediction = ''.join(chars)
            
            # Format với dấu gạch ngang
            formatted_prediction = format_license_plate(prediction)
            
            print(f"🔤 Raw prediction: '{prediction}'")
            print(f"📋 Formatted prediction: '{formatted_prediction}'")
            
            return formatted_prediction, outputs.get('attention_weights')
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return "", None


def visualize_result(original_image, prediction, save_path=None):
    """
    Hiển thị kết quả dự đoán
    
    Args:
        original_image: Ảnh gốc
        prediction: Kết quả dự đoán
        save_path: Đường dẫn lưu hình (optional)
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Hiển thị ảnh gốc
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title(f'Input Image\nShape: {original_image.shape}', fontsize=12)
        axes[0].axis('off')
        
        # Hiển thị kết quả
        axes[1].text(0.5, 0.7, 'License Plate Prediction:', ha='center', va='center', 
                    fontsize=16, fontweight='bold', transform=axes[1].transAxes)
        axes[1].text(0.5, 0.4, f"'{prediction}'", ha='center', va='center', 
                    fontsize=24, color='red', fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
                    transform=axes[1].transAxes)
        
        # Thêm thông tin
        axes[1].text(0.5, 0.1, f"Length: {len(prediction)} characters", 
                    ha='center', va='center', fontsize=10, 
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


def main():
    """
    Hàm main
    """
    parser = argparse.ArgumentParser(description='Test VOCR Model with Single Image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save result image')
    
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
    
    # Check files exist
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Load model
    print(f"📂 Loading model from: {args.checkpoint}")
    model = load_trained_model(args.checkpoint, config)
    
    if model is None:
        print("❌ Failed to load model. Exiting...")
        return
    
    model = model.to(device)
    
    print(f"🖼️  Testing image: {args.image}")
    
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
    print(f"\n🎯 FINAL RESULT")
    print("="*50)
    print(f"📷 Input: {os.path.basename(args.image)}")
    print(f"🔤 Prediction: '{prediction}'")
    print(f"📏 Length: {len(prediction)} characters")
    
    # Analyze prediction
    if prediction:
        if '-' in prediction:
            parts = prediction.split('-')
            if len(parts) == 2:
                print(f"🏷️  Analysis:")
                print(f"   Province + Series: {parts[0]}")
                print(f"   Number: {parts[1]}")
        else:
            print("⚠️  Missing dash in license plate format")
    else:
        print("❌ No prediction generated")
    
    print("="*50)
    
    # Visualize if possible
    try:
        print("\n📊 Creating visualization...")
        save_path = args.save if args.save else None
        visualize_result(original_image, prediction, save_path)
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    print("\n✅ Testing completed!")


if __name__ == '__main__':
    main()