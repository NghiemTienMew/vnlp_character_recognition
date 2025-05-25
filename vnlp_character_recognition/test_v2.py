"""
test_v2.py - Improved test script với proper weight loading
"""
import os
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path

# Setup path
root_path = Path(__file__).parent
sys.path.append(str(root_path))

from config.config import get_config
from models.vocr import VOCR


def load_model_weights_only(weights_path):
    """
    Load chỉ model weights (không có optimizer, scheduler)
    
    Args:
        weights_path: Đường dẫn model weights
        
    Returns:
        model: Model đã load weights
    """
    try:
        print(f"📂 Loading model weights from: {weights_path}")
        
        # Tạo model
        model = VOCR()
        
        # Load weights
        weights = torch.load(weights_path, map_location='cpu')
        
        # Load state dict
        model.load_state_dict(weights)
        model.eval()
        
        print(f"✅ Model weights loaded successfully!")
        print(f"📊 Model info: {model.get_model_info()}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return None


def extract_weights_from_checkpoint(checkpoint_path, output_path):
    """
    Extract weights từ checkpoint nếu cần
    """
    try:
        print(f"🔧 Extracting weights from checkpoint...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            weights = checkpoint['model_state_dict']
            torch.save(weights, output_path)
            print(f"✅ Weights extracted to: {output_path}")
            return True
        else:
            print(f"❌ No model_state_dict found in checkpoint")
            return False
            
    except Exception as e:
        print(f"❌ Error extracting weights: {e}")
        return False


def preprocess_image(image_path, target_size=(32, 140)):
    """
    Preprocess image for model
    """
    try:
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        print(f"📷 Original shape: {image.shape}")
        
        # Resize
        image = cv2.resize(image, (target_size[1], target_size[0]))
        print(f"🔄 Resized to: {image.shape}")
        
        # Normalize to [-1, 1]
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        
        # Convert to tensor [1, 1, H, W]
        tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        
        print(f"✅ Tensor shape: {tensor.shape}")
        print(f"📊 Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        
        return tensor, image
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return None, None


def predict_license_plate(model, image_tensor, config, device='cuda'):
    """
    Predict license plate
    """
    try:
        model = model.to(device)
        image_tensor = image_tensor.to(device)
        
        print(f"🧠 Running inference on {device}...")
        
        with torch.no_grad():
            # Forward pass
            outputs = model(image_tensor, targets=None, teacher_forcing_ratio=0.0)
            logits = outputs['outputs']  # [1, seq_len, num_classes]
            
            print(f"📊 Output shape: {logits.shape}")
            
            # Get predictions
            _, predicted_indices = torch.max(logits, dim=2)
            indices = predicted_indices[0].cpu().tolist()
            
            # Decode
            chars = []
            idx_to_char = config.get('data.idx_to_char', {})
            
            print(f"🔢 Predicted indices: {indices}")
            print(f"🔤 Decoding:")
            
            for idx in indices:
                if idx == 0:  # <pad>
                    continue
                elif idx == 1:  # <sos>
                    continue
                elif idx == 2:  # <eos>
                    break
                elif idx in idx_to_char:
                    char = idx_to_char[idx]
                    chars.append(char)
                    print(f"   {idx} -> '{char}'")
                else:
                    print(f"   {idx} -> UNKNOWN")
            
            prediction = ''.join(chars)
            
            # Format với dấu gạch ngang
            formatted = format_license_plate(prediction)
            
            print(f"🔤 Raw: '{prediction}'")
            print(f"📋 Formatted: '{formatted}'")
            
            return formatted
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return ""


def format_license_plate(text):
    """Format license plate với dấu gạch ngang"""
    if not text:
        return text
    
    text = text.upper().strip()
    
    if '-' in text:
        return text
    
    # Find last letter position
    last_letter_pos = -1
    for i, char in enumerate(text):
        if char.isalpha():
            last_letter_pos = i
    
    # Add dash after last letter
    if last_letter_pos >= 0 and last_letter_pos < len(text) - 1:
        return text[:last_letter_pos + 1] + '-' + text[last_letter_pos + 1:]
    
    return text


def main():
    parser = argparse.ArgumentParser(description='Test VOCR Model v2')
    parser.add_argument('--image', required=True, help='Image path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--extract', action='store_true', help='Extract weights from checkpoint')
    
    args = parser.parse_args()
    
    print("🚀 VOCR Model Testing v2")
    print("="*50)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    # Load config
    config = get_config()
    
    # Check files
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Handle weights loading
    weights_path = args.checkpoint
    
    if args.extract or args.checkpoint.endswith('.pth'):
        # Try to extract weights if it's a full checkpoint
        weights_extracted_path = args.checkpoint.replace('.pth', '_weights_only.pth')
        
        if not os.path.exists(weights_extracted_path):
            print("🔧 Extracting weights from checkpoint...")
            if extract_weights_from_checkpoint(args.checkpoint, weights_extracted_path):
                weights_path = weights_extracted_path
            else:
                print("⚠️  Using original checkpoint...")
        else:
            print(f"✅ Using existing extracted weights: {weights_extracted_path}")
            weights_path = weights_extracted_path
    
    # Load model
    model = load_model_weights_only(weights_path)
    if model is None:
        print("❌ Failed to load model!")
        return
    
    # Preprocess image
    print(f"\n🖼️  Processing image: {os.path.basename(args.image)}")
    image_tensor, original_image = preprocess_image(args.image)
    
    if image_tensor is None:
        print("❌ Failed to preprocess image!")
        return
    
    # Predict
    print(f"\n🧠 Making prediction...")
    prediction = predict_license_plate(model, image_tensor, config, device)
    
    # Results
    print(f"\n🎯 RESULTS")
    print("="*50)
    print(f"📷 Image: {os.path.basename(args.image)}")
    print(f"🔤 Prediction: '{prediction}'")
    print(f"📏 Length: {len(prediction)}")
    
    # Expected result for validation
    expected = "36A-02391"
    if prediction == expected:
        print(f"✅ CORRECT! Matches expected: {expected}")
    else:
        print(f"❌ INCORRECT! Expected: {expected}")
        
        # Character-by-character comparison
        print(f"\n📝 Character comparison:")
        print(f"   Predicted: {list(prediction)}")
        print(f"   Expected:  {list(expected)}")
    
    print("="*50)


if __name__ == '__main__':
    main()