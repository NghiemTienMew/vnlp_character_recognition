"""
visualize.py và checkpoints.py - Các utility functions cần thiết
"""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Any
import seaborn as sns
from pathlib import Path


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, filepath):
    """
    Lưu checkpoint
    
    Args:
        model: Model để lưu
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Epoch hiện tại
        val_loss: Validation loss
        filepath: Đường dẫn file để lưu
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': val_loss
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
        
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def load_checkpoint(filepath):
    """
    Tải checkpoint
    
    Args:
        filepath: Đường dẫn file checkpoint
        
    Returns:
        Dict: Checkpoint data
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float], 
                        accuracies: List[float],
                        save_path: Optional[str] = None):
    """
    Vẽ đường cong training
    
    Args:
        train_losses: Danh sách train loss
        val_losses: Danh sách validation loss
        accuracies: Danh sách accuracy
        save_path: Đường dẫn để lưu hình
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(epochs, accuracies, 'g-', label='Accuracy', linewidth=2)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"Error plotting training curves: {e}")


def visualize_predictions(images: torch.Tensor,
                         predictions: List[str],
                         targets: List[str],
                         save_path: Optional[str] = None,
                         max_samples: int = 8):
    """
    Visualize dự đoán
    
    Args:
        images: Tensor ảnh
        predictions: Danh sách dự đoán
        targets: Danh sách target
        save_path: Đường dẫn lưu hình
        max_samples: Số lượng samples tối đa để hiển thị
    """
    try:
        batch_size = min(images.size(0), max_samples, len(predictions), len(targets))
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(batch_size):
            if i >= len(axes):
                break
                
            # Chuyển đổi tensor thành numpy array
            if images.dim() == 4:  # [batch, channel, height, width]
                img = images[i, 0].cpu().numpy()  # Lấy channel đầu tiên
            else:
                img = images[i].cpu().numpy()
            
            # Hiển thị ảnh
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Pred: {predictions[i]}\nTrue: {targets[i]}')
            axes[i].axis('off')
        
        # Ẩn các subplot không dùng
        for i in range(batch_size, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions visualization saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing predictions: {e}")


def plot_attention_weights(attention_weights: torch.Tensor,
                          input_sequence: str,
                          output_sequence: str,
                          save_path: Optional[str] = None):
    """
    Visualize attention weights
    
    Args:
        attention_weights: Tensor attention weights [seq_len, seq_len]
        input_sequence: Chuỗi input
        output_sequence: Chuỗi output
        save_path: Đường dẫn lưu hình
    """
    try:
        if attention_weights is None:
            print("No attention weights to visualize")
            return
        
        # Chuyển đổi thành numpy
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Nếu có batch dimension, lấy sample đầu tiên
        if attention_weights.ndim == 3:
            attention_weights = attention_weights[0]
        
        plt.figure(figsize=(10, 8))
        
        # Tạo heatmap
        sns.heatmap(attention_weights, 
                   xticklabels=list(input_sequence),
                   yticklabels=list(output_sequence),
                   cmap='Blues',
                   cbar=True)
        
        plt.title('Attention Weights')
        plt.xlabel('Input Sequence')
        plt.ylabel('Output Sequence')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention visualization saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"Error plotting attention weights: {e}")


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """
    Tạo thư mục thí nghiệm
    
    Args:
        base_dir: Thư mục gốc
        experiment_name: Tên thí nghiệm
        
    Returns:
        str: Đường dẫn thư mục thí nghiệm
    """
    try:
        experiment_dir = os.path.join(base_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Tạo các thư mục con
        subdirs = ['checkpoints', 'logs', 'visualizations', 'predictions']
        for subdir in subdirs:
            os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
        
        return experiment_dir
        
    except Exception as e:
        print(f"Error creating experiment directory: {e}")
        return base_dir


def log_model_summary(model, logger):
    """
    Log thông tin tóm tắt model
    
    Args:
        model: Model
        logger: Logger
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info("="*50)
        logger.info("MODEL SUMMARY")
        logger.info("="*50)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Log model info nếu có
        if hasattr(model, 'get_model_info'):
            model_info = model.get_model_info()
            for key, value in model_info.items():
                logger.info(f"{key}: {value}")
        
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Error logging model summary: {e}")


def save_predictions_to_file(predictions: List[str], 
                           targets: List[str],
                           image_paths: List[str],
                           save_path: str):
    """
    Lưu predictions vào file
    
    Args:
        predictions: Danh sách dự đoán
        targets: Danh sách target
        image_paths: Danh sách đường dẫn ảnh
        save_path: Đường dẫn file để lưu
    """
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Image Path\tTarget\tPrediction\tCorrect\n")
            
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                img_path = image_paths[i] if i < len(image_paths) else f"image_{i}"
                is_correct = "YES" if pred == target else "NO"
                f.write(f"{img_path}\t{target}\t{pred}\t{is_correct}\n")
        
        print(f"Predictions saved to {save_path}")
        
    except Exception as e:
        print(f"Error saving predictions: {e}")


def calculate_model_size(model):
    """
    Tính kích thước model
    
    Args:
        model: Model
        
    Returns:
        Dict: Thông tin kích thước model
    """
    try:
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            'param_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024,
            'total_size_mb': total_size / 1024 / 1024
        }
        
    except Exception as e:
        print(f"Error calculating model size: {e}")
        return {'param_size_mb': 0, 'buffer_size_mb': 0, 'total_size_mb': 0}