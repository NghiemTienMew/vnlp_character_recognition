"""
test_checkpoint.py - Script test checkpoint đã tải về
"""
import os
import sys
import torch
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Thêm đường dẫn root vào sys.path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from config.config import get_config, load_config
from models.vocr import VOCR
from data.preprocessor import Preprocessor
from data.dataset import create_data_loaders
from utils.metrics import compute_metrics


def format_vietnamese_license_plate(prediction: str) -> str:
    """
    Format prediction thành biển số VN chuẩn
    Input: '12C06406'
    Output: '12C-06406'
    """
    prediction = prediction.strip()
    
    # Pattern biển số VN: 2 số + 1-2 chữ + 5 số
    if len(prediction) >= 7:
        # Tìm vị trí chữ cái cuối cùng
        last_letter_pos = -1
        for i, char in enumerate(prediction):
            if char.isalpha():
                last_letter_pos = i
        
        if last_letter_pos >= 2:  # Có ít nhất 2 số ở đầu và có chữ
            formatted = prediction[:last_letter_pos+1] + '-' + prediction[last_letter_pos+1:]
            return formatted
    
    return prediction


def load_model_from_checkpoint(checkpoint_path: str, device):
    """
    Load model từ checkpoint
    
    Args:
        checkpoint_path: Đường dẫn đến checkpoint
        device: Device để load model
        
    Returns:
        model: VOCR model đã load weights
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Tạo model
    model = VOCR().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Checkpoint info:")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocess một ảnh đơn lẻ
    
    Args:
        image_path: Đường dẫn đến ảnh
        
    Returns:
        torch.Tensor: Ảnh đã preprocess
    """
    try:
        # Đọc ảnh
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Resize về kích thước chuẩn
        image = cv2.resize(image, (140, 32))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        
        # Convert to tensor và thêm batch dimension
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # [1, 1, 32, 140]
        
        return image_tensor
        
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None


def predict_single_image(model, image_path: str, device):
    """
    Predict biển số cho một ảnh
    
    Args:
        model: VOCR model
        image_path: Đường dẫn ảnh
        device: Device
        
    Returns:
        str: Biển số predicted
    """
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        return "ERROR"
    
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        try:
            predictions = model.predict(image_tensor)
            raw_prediction = predictions[0] if len(predictions) > 0 else ""
            
            # Format thành biển số VN chuẩn
            formatted_prediction = format_vietnamese_license_plate(raw_prediction)
            
            return formatted_prediction
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "ERROR"


def test_on_dataset(model, device, max_samples=100):
    """
    Test model trên dataset
    
    Args:
        model: VOCR model
        device: Device
        max_samples: Số lượng samples tối đa để test
    """
    print(f"\nTesting model on dataset (max {max_samples} samples)...")
    
    try:
        # Tạo data loaders
        _, val_loader, _ = create_data_loaders()
        
        all_predictions = []
        all_targets = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                if len(all_predictions) >= max_samples:
                    break
                
                try:
                    # Get batch data
                    if len(batch_data) == 3:
                        images, targets, labels = batch_data
                    else:
                        images, targets = batch_data
                        labels = None
                    
                    # Move to device
                    images = images.to(device)
                    
                    # Predict
                    predictions = model.predict(images)
                    
                    # Format predictions
                    formatted_predictions = [format_vietnamese_license_plate(pred) for pred in predictions]
                    
                    # Get target strings
                    if labels is not None:
                        target_strings = [str(label) for label in labels]
                    else:
                        # Decode from tensor (simplified)
                        target_strings = ["UNKNOWN"] * len(predictions)
                    
                    # Collect results
                    all_predictions.extend(formatted_predictions)
                    all_targets.extend(target_strings)
                    
                    # Print some examples
                    if batch_idx == 0:
                        print(f"\nSample predictions:")
                        for i in range(min(5, len(formatted_predictions))):
                            match_status = "✓" if formatted_predictions[i] == target_strings[i] else "✗"
                            print(f"  {match_status} Pred: '{formatted_predictions[i]}' | Target: '{target_strings[i]}'")
                
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
        
        # Calculate metrics
        if len(all_predictions) > 0 and len(all_targets) > 0:
            try:
                metrics = compute_metrics(all_predictions, all_targets)
                
                print(f"\nTest Results on {len(all_predictions)} samples:")
                print(f"  - Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
                print(f"  - Character Accuracy: {metrics['character_accuracy']:.4f} ({metrics['character_accuracy']*100:.2f}%)")
                print(f"  - Edit Distance: {metrics['edit_distance']:.4f}")
                
                # Show some correct predictions
                correct_preds = [(p, t) for p, t in zip(all_predictions, all_targets) if p == t]
                if len(correct_preds) > 0:
                    print(f"\nSome correct predictions:")
                    for i, (pred, target) in enumerate(correct_preds[:5]):
                        print(f"  ✓ '{pred}'")
                
            except Exception as e:
                print(f"Error calculating metrics: {e}")
        
    except Exception as e:
        print(f"Error testing on dataset: {e}")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Test VOCR Checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (.pth)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image to test')
    parser.add_argument('--dataset', action='store_true',
                       help='Test on validation dataset')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device to use')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config()
    
    # Setup device
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Check checkpoint file
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    print(f"Loading model from checkpoint...")
    
    try:
        # Load model
        model = load_model_from_checkpoint(args.checkpoint, device)
        
        # Test single image
        if args.image:
            if os.path.exists(args.image):
                print(f"\nTesting single image: {args.image}")
                prediction = predict_single_image(model, args.image, device)
                print(f"Predicted license plate: '{prediction}'")
            else:
                print(f"Error: Image file not found: {args.image}")
        
        # Test on dataset
        if args.dataset:
            test_on_dataset(model, device, max_samples=200)
        
        # If no specific test requested, run dataset test
        if not args.image and not args.dataset:
            print("No specific test requested. Running dataset test...")
            test_on_dataset(model, device, max_samples=50)
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()