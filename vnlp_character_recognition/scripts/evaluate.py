"""
evaluate.py - Script đánh giá mô hình VOCR
"""
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import get_config, load_config
from data.dataset import create_data_loaders
from models.vocr import VOCR
from losses.combined_loss import create_loss_function
from utils.metrics import compute_metrics, log_incorrect_predictions, visualize_confusion_matrix
from utils.checkpoints import load_checkpoint
from utils.visualize import visualize_predictions, visualize_attention


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate VOCR model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    
    return parser.parse_args()


def evaluate(model, data_loader, device, save_dir, num_samples=10):
    """
    Đánh giá mô hình trên tập dữ liệu
    
    Args:
        model: Mô hình cần đánh giá
        data_loader: DataLoader cho dữ liệu đánh giá
        device: Device để chạy model
        save_dir: Thư mục lưu kết quả
        num_samples: Số lượng mẫu để trực quan hóa
        
    Returns:
        Dict: Dictionary chứa các metrics
    """
    model.eval()
    
    # Khởi tạo các danh sách kết quả
    all_predictions = []
    all_targets = []
    selected_images = []
    selected_predictions = []
    selected_targets = []
    
    # Tqdm progress bar
    pbar = tqdm(data_loader, desc='Evaluating')
    
    # Đo thời gian xử lý
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, (images, targets, original_labels) in enumerate(pbar):
            images = images.to(device)
            
            # Đo thời gian xử lý
            start_time = time.time()
            
            # Forward pass
            outputs = model(images)
            
            # Kết thúc đo thời gian
            end_time = time.time()
            batch_time = end_time - start_time
            inference_times.append(batch_time / images.size(0))  # Thời gian trung bình cho mỗi ảnh
            
            # Get predictions
            predictions = model.predict(images)
            
            # Lưu kết quả
            all_predictions.extend(predictions)
            all_targets.extend(original_labels)
            
            # Lưu một số mẫu cho trực quan hóa
            if len(selected_images) < num_samples:
                # Chọn số lượng mẫu cần thêm
                num_to_add = min(num_samples - len(selected_images), images.size(0))
                
                # Lưu các mẫu
                selected_images.extend(images[:num_to_add].cpu())
                selected_predictions.extend(predictions[:num_to_add])
                selected_targets.extend(original_labels[:num_to_add])
    
    # Tính metrics
    metrics = compute_metrics(all_predictions, all_targets)
    
    # Thêm thông tin về thời gian xử lý
    metrics['avg_inference_time'] = sum(inference_times) / len(inference_times)
    metrics['fps'] = 1.0 / metrics['avg_inference_time']
    
    # In metrics
    print(f"Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Character Accuracy: {metrics['character_accuracy']:.4f}")
    print(f"Edit Distance: {metrics['edit_distance']:.4f}")
    print(f"Normalized Edit Distance: {metrics['normalized_edit_distance']:.4f}")
    print(f"Average Inference Time: {metrics['avg_inference_time'] * 1000:.2f} ms")
    print(f"FPS: {metrics['fps']:.2f}")
    
    # Tạo thư mục kết quả
    os.makedirs(save_dir, exist_ok=True)
    
    # Lưu metrics
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        for key, value in metrics.items():
            if key not in ['confusion_matrix', 'confusion_matrix_labels']:
                f.write(f"{key}: {value}\n")
    
    # Lưu các dự đoán sai
    incorrect_predictions = log_incorrect_predictions(
        all_predictions, all_targets, save_path=os.path.join(save_dir, 'incorrect_predictions.txt')
    )
    
    # Lưu ma trận nhầm lẫn nếu có
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        labels = metrics['confusion_matrix_labels']
        visualize_confusion_matrix(
            cm, labels, save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
    
    # Trực quan hóa các dự đoán
    if selected_images:
        selected_images_tensor = torch.stack(selected_images)
        fig = visualize_predictions(selected_images_tensor, selected_predictions, selected_targets)
        plt.savefig(os.path.join(save_dir, 'sample_predictions.png'))
        plt.close(fig)
        
        # Trực quan hóa attention nếu có
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'attention'):
            for i in range(min(5, len(selected_images))):
                # Forward lại để lấy attention weights
                image = selected_images[i].unsqueeze(0).to(device)
                outputs = model(image)
                
                if 'attention_weights' in outputs and outputs['attention_weights'] is not None:
                    attn_weights = outputs['attention_weights'][0].cpu().numpy()
                    fig = visualize_attention(
                        selected_images[i], 
                        attn_weights, 
                        selected_predictions[i], 
                        selected_targets[i]
                    )
                    plt.savefig(os.path.join(save_dir, f'attention_sample_{i+1}.png'))
                    plt.close(fig)
    
    return metrics


def main():
    """
    Main evaluation function
    """
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = get_config()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
        print("Warning: CUDA is not available. Using CPU instead.")
    
    # Create data loaders
    _, _, test_loader = create_data_loaders()
    
    # Create model
    model = VOCR().to(device)
    
    # Load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise ValueError(f"No checkpoint found at {args.checkpoint}")
    
    checkpoint = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from {args.checkpoint} (epoch {checkpoint['epoch']})")
    
    # Create save directory
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config
    config.save(os.path.join(save_dir, 'config.yaml'))
    
    # Evaluate model
    metrics = evaluate(model, test_loader, device, save_dir, args.num_samples)
    
    print(f"Evaluation completed. Results saved to {save_dir}")


if __name__ == '__main__':
    main()