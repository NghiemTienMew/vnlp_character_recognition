"""
visualize.py - Công cụ trực quan hóa cho mô hình VOCR
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from typing import List, Tuple, Optional
import os


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """
    Khôi phục ảnh từ dạng tensor đã chuẩn hóa về dạng numpy array
    
    Args:
        image: Tensor ảnh, kích thước [channels, height, width]
        
    Returns:
        np.ndarray: Ảnh dạng numpy array, kích thước [height, width, channels]
    """
    # Chuyển về CPU nếu cần
    if image.is_cuda:
        image = image.cpu()
    
    # Chuyển sang numpy array
    image_np = image.numpy()
    
    # Khôi phục chuẩn hóa (giả sử đã chuẩn hóa với mean=0.5, std=0.5)
    image_np = image_np * 0.5 + 0.5
    
    # Đảm bảo các giá trị nằm trong khoảng [0, 1]
    image_np = np.clip(image_np, 0, 1)
    
    # Nếu là ảnh grayscale (1 kênh)
    if image_np.shape[0] == 1:
        image_np = image_np.squeeze(0)  # [height, width]
        # Chuyển về RGB để hiển thị
        image_np = np.stack([image_np, image_np, image_np], axis=2)  # [height, width, 3]
    else:
        # Chuyển từ [channels, height, width] sang [height, width, channels]
        image_np = np.transpose(image_np, (1, 2, 0))
    
    # Chuyển về dạng uint8 (0-255)
    image_np = (image_np * 255).astype(np.uint8)
    
    return image_np


def visualize_predictions(images: torch.Tensor, predictions: List[str], targets: List[str], max_samples: int = 8) -> plt.Figure:
    """
    Trực quan hóa dự đoán
    
    Args:
        images: Tensor ảnh, kích thước [batch_size, channels, height, width]
        predictions: Danh sách các chuỗi dự đoán
        targets: Danh sách các chuỗi thực tế
        max_samples: Số lượng mẫu tối đa để hiển thị
        
    Returns:
        plt.Figure: Figure chứa kết quả trực quan hóa
    """
    num_samples = min(len(images), max_samples)
    
    # Tạo figure
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # Lấy ảnh và chuyển về dạng numpy array
        image_np = denormalize_image(images[i])
        
        # Hiển thị ảnh
        axes[i].imshow(image_np)
        
        # Thêm nhãn
        title = f"Pred: {predictions[i]}, Target: {targets[i]}"
        if predictions[i] == targets[i]:
            title += " ✓"
        else:
            title += " ✗"
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    return fig


def visualize_attention(image: torch.Tensor, attention_weights: np.ndarray, 
                       prediction: str, target: str) -> plt.Figure:
    """
    Trực quan hóa cơ chế attention
    
    Args:
        image: Tensor ảnh, kích thước [channels, height, width]
        attention_weights: Ma trận trọng số attention, kích thước [seq_len, encoder_seq_len]
        prediction: Chuỗi dự đoán
        target: Chuỗi thực tế
        
    Returns:
        plt.Figure: Figure chứa kết quả trực quan hóa
    """
    # Lấy ảnh và chuyển về dạng numpy array
    image_np = denormalize_image(image)
    
    # Xác định kích thước của figure
    seq_len = min(len(prediction) + 2, attention_weights.shape[0])  # +2 cho <sos> và <eos>
    
    fig, axes = plt.subplots(seq_len, 1, figsize=(10, 2 * seq_len))
    if seq_len == 1:
        axes = [axes]
    
    # Hiển thị ảnh gốc
    axes[0].imshow(image_np)
    axes[0].set_title(f"Input Image - Pred: {prediction}, Target: {target}")
    axes[0].axis('off')
    
    # Hiển thị attention map cho mỗi ký tự
    characters = ['<sos>'] + list(prediction) + ['<eos>']
    
    for t in range(1, seq_len):
        # Lấy attention weights cho ký tự tại vị trí t
        attn = attention_weights[t-1]
        
        # Đảm bảo attention weights có kích thước phù hợp với ảnh
        h, w = image_np.shape[:2]
        
        # Tính kích thước của feature map
        feature_h = attention_weights.shape[1] // (attention_weights.shape[1] // h)
        feature_w = attention_weights.shape[1] - feature_h
        
        # Reshape attention weights thành 2D map
        attn_map = attn[:feature_h * feature_w].reshape(feature_h, feature_w)
        
        # Resize attention map về kích thước của ảnh
        attn_map = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Chuẩn hóa để hiển thị
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        # Tạo heatmap
        heatmap = cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Trộn heatmap với ảnh gốc
        alpha = 0.5
        overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
        
        # Hiển thị
        axes[t].imshow(overlay)
        axes[t].set_title(f"Attention for '{characters[t-1]}'")
        axes[t].axis('off')
    
    plt.tight_layout()
    
    return fig


def visualize_training_history(history: dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Trực quan hóa lịch sử huấn luyện
    
    Args:
        history: Dictionary chứa lịch sử huấn luyện
        save_path: Đường dẫn để lưu hình ảnh
        
    Returns:
        plt.Figure: Figure chứa kết quả trực quan hóa
    """
    # Tạo figure với 2 subplots (loss và accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['accuracy'], label='Accuracy')
    ax2.plot(history['character_accuracy'], label='Character Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Metrics')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Lưu hình ảnh nếu cần
    if save_path:
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_learning_rate(history: dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Trực quan hóa học tốc độ học
    
    Args:
        history: Dictionary chứa lịch sử huấn luyện
        save_path: Đường dẫn để lưu hình ảnh
        
    Returns:
        plt.Figure: Figure chứa kết quả trực quan hóa
    """
    if 'learning_rate' not in history:
        return None
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot learning rate
    ax.plot(history['learning_rate'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True)
    
    # Đặt trục y theo logarithm để dễ quan sát
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Lưu hình ảnh nếu cần
    if save_path:
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig