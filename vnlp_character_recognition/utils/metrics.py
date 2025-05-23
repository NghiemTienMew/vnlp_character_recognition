"""
metrics.py - Các metrics đánh giá cho mô hình VOCR
"""
import numpy as np
import Levenshtein
from typing import List, Dict, Tuple
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config.config import get_config


def compute_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Tính độ chính xác (toàn bộ chuỗi phải giống nhau)
    
    Args:
        predictions: Danh sách các chuỗi dự đoán
        targets: Danh sách các chuỗi thực tế
        
    Returns:
        float: Độ chính xác
    """
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(targets) if len(targets) > 0 else 0.0


def compute_character_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Tính độ chính xác ký tự (tỷ lệ ký tự đúng)
    
    Args:
        predictions: Danh sách các chuỗi dự đoán
        targets: Danh sách các chuỗi thực tế
        
    Returns:
        float: Độ chính xác ký tự
    """
    total_chars = 0
    correct_chars = 0
    
    for pred, target in zip(predictions, targets):
        # Chuẩn hóa độ dài
        max_len = max(len(pred), len(target))
        pred_padded = pred.ljust(max_len)
        target_padded = target.ljust(max_len)
        
        # Đếm số ký tự đúng
        for p_char, t_char in zip(pred_padded, target_padded):
            if p_char == t_char:
                correct_chars += 1
            total_chars += 1
    
    return correct_chars / total_chars if total_chars > 0 else 0.0


def compute_edit_distance(predictions: List[str], targets: List[str]) -> float:
    """
    Tính khoảng cách Levenshtein trung bình
    
    Args:
        predictions: Danh sách các chuỗi dự đoán
        targets: Danh sách các chuỗi thực tế
        
    Returns:
        float: Khoảng cách Levenshtein trung bình
    """
    distances = [Levenshtein.distance(p, t) for p, t in zip(predictions, targets)]
    return sum(distances) / len(targets) if len(targets) > 0 else 0.0


def compute_normalized_edit_distance(predictions: List[str], targets: List[str]) -> float:
    """
    Tính khoảng cách Levenshtein chuẩn hóa trung bình
    
    Args:
        predictions: Danh sách các chuỗi dự đoán
        targets: Danh sách các chuỗi thực tế
        
    Returns:
        float: Khoảng cách Levenshtein chuẩn hóa trung bình
    """
    normalized_distances = []
    
    for p, t in zip(predictions, targets):
        if len(t) == 0:
            normalized_distances.append(1.0 if len(p) > 0 else 0.0)
        else:
            normalized_distances.append(Levenshtein.distance(p, t) / len(t))
    
    return sum(normalized_distances) / len(targets) if len(targets) > 0 else 0.0


def compute_confusion_matrix(predictions: List[str], targets: List[str]) -> np.ndarray:
    """
    Tính ma trận nhầm lẫn giữa các ký tự
    
    Args:
        predictions: Danh sách các chuỗi dự đoán
        targets: Danh sách các chuỗi thực tế
        
    Returns:
        np.ndarray: Ma trận nhầm lẫn
    """
    config = get_config()
    chars = config.get('data.chars')
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    
    # Lấy tất cả các ký tự trong predictions và targets
    all_chars = []
    for string in predictions + targets:
        all_chars.extend(list(string))
    
    # Lấy các ký tự duy nhất
    unique_chars = sorted(set(all_chars))
    
    # Chuẩn bị dữ liệu cho ma trận nhầm lẫn
    pred_chars = []
    target_chars = []
    
    for pred, target in zip(predictions, targets):
        # Căn chỉnh chuỗi
        max_len = max(len(pred), len(target))
        pred_padded = pred.ljust(max_len)
        target_padded = target.ljust(max_len)
        
        # Thêm ký tự
        for p_char, t_char in zip(pred_padded, target_padded):
            pred_chars.append(p_char)
            target_chars.append(t_char)
    
    # Tính ma trận nhầm lẫn
    cm = confusion_matrix(target_chars, pred_chars, labels=unique_chars)
    
    return cm, unique_chars


def visualize_confusion_matrix(cm: np.ndarray, labels: List[str], save_path: str = None):
    """
    Vẽ ma trận nhầm lẫn
    
    Args:
        cm: Ma trận nhầm lẫn
        labels: Nhãn cho các hàng và cột
        save_path: Đường dẫn để lưu hình ảnh
    """
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    mask = np.zeros_like(cm_normalized)
    mask[cm_normalized < 0.01] = True
    
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, 
                     yticklabels=labels, mask=mask, cbar=False)
    
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_metrics(predictions: List[str], targets: List[str]) -> Dict:
    """
    Tính toán tất cả các metrics
    
    Args:
        predictions: Danh sách các chuỗi dự đoán
        targets: Danh sách các chuỗi thực tế
        
    Returns:
        Dict: Dictionary chứa các metrics
    """
    # Kiểm tra tính hợp lệ của đầu vào
    if len(predictions) != len(targets):
        raise ValueError("Length of predictions and targets must be the same")
    
    if len(predictions) == 0:
        return {
            'accuracy': 0.0,
            'character_accuracy': 0.0,
            'edit_distance': 0.0,
            'normalized_edit_distance': 0.0
        }
    
    # Tính các metrics
    accuracy = compute_accuracy(predictions, targets)
    character_accuracy = compute_character_accuracy(predictions, targets)
    edit_distance = compute_edit_distance(predictions, targets)
    normalized_edit_distance = compute_normalized_edit_distance(predictions, targets)
    
    # Tính ma trận nhầm lẫn nếu cần
    config = get_config()
    if config.get('evaluation.confusion_matrix', True):
        cm, labels = compute_confusion_matrix(predictions, targets)
    else:
        cm, labels = None, None
    
    # Tạo dictionary kết quả
    metrics = {
        'accuracy': accuracy,
        'character_accuracy': character_accuracy,
        'edit_distance': edit_distance,
        'normalized_edit_distance': normalized_edit_distance
    }
    
    if cm is not None:
        metrics['confusion_matrix'] = cm
        metrics['confusion_matrix_labels'] = labels
    
    return metrics


def log_incorrect_predictions(predictions: List[str], targets: List[str], 
                             images: torch.Tensor = None, save_path: str = None):
    """
    Ghi lại các dự đoán sai
    
    Args:
        predictions: Danh sách các chuỗi dự đoán
        targets: Danh sách các chuỗi thực tế
        images: Tensor chứa các ảnh tương ứng
        save_path: Đường dẫn để lưu kết quả
    """
    incorrect = []
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        if pred != target:
            item = {
                'id': i,
                'prediction': pred,
                'target': target,
                'edit_distance': Levenshtein.distance(pred, target)
            }
            incorrect.append(item)
    
    # Sắp xếp theo khoảng cách Levenshtein giảm dần
    incorrect = sorted(incorrect, key=lambda x: x['edit_distance'], reverse=True)
    
    # Ghi ra file
    if save_path:
        with open(save_path, 'w') as f:
            f.write(f"Total incorrect predictions: {len(incorrect)}/{len(predictions)} "
                   f"({len(incorrect)/len(predictions)*100:.2f}%)\n\n")
            
            for item in incorrect:
                f.write(f"ID: {item['id']}\n")
                f.write(f"Prediction: {item['prediction']}\n")
                f.write(f"Target: {item['target']}\n")
                f.write(f"Edit Distance: {item['edit_distance']}\n")
                f.write("\n")
    
    # Trả về danh sách các dự đoán sai
    return incorrect