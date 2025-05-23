"""
checkpoints.py - Lưu và nạp checkpoints
"""
import os
import torch
from typing import Dict, Optional


def save_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                   epoch: int, loss: float, best_loss: float, path: str) -> None:
    """
    Lưu checkpoint của mô hình
    
    Args:
        model: Mô hình cần lưu
        optimizer: Optimizer cần lưu
        epoch: Epoch hiện tại
        loss: Loss hiện tại
        best_loss: Loss tốt nhất từ trước đến nay
        path: Đường dẫn để lưu checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'best_loss': best_loss
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Lưu checkpoint
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """
    Nạp checkpoint của mô hình
    
    Args:
        path: Đường dẫn để nạp checkpoint
        model: Mô hình cần nạp trọng số
        optimizer: Optimizer cần nạp trạng thái (có thể None)
        
    Returns:
        Dict: Dictionary chứa thông tin của checkpoint
    """
    # Nạp checkpoint
    checkpoint = torch.load(path, map_location=next(model.parameters()).device)
    
    # Nạp trọng số cho mô hình
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Nạp trạng thái cho optimizer nếu có
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def generate_model_summary(model: torch.nn.Module) -> str:
    """
    Tạo bản tóm tắt của mô hình
    
    Args:
        model: Mô hình cần tóm tắt
        
    Returns:
        str: Chuỗi chứa bản tóm tắt
    """
    summary = []
    
    # Thêm thông tin về kiến trúc
    summary.append("Model Architecture:")
    summary.append("-" * 60)
    
    total_params = 0
    trainable_params = 0
    
    # Duyệt qua các module con
    for name, parameter in model.named_parameters():
        param_count = parameter.numel()
        total_params += param_count
        
        if parameter.requires_grad:
            trainable_params += param_count
        
        summary.append(f"{name}: {list(parameter.shape)}, Params: {param_count}, "
                       f"Trainable: {parameter.requires_grad}")
    
    # Thêm thông tin tổng quan
    summary.append("-" * 60)
    summary.append(f"Total Parameters: {total_params:,}")
    summary.append(f"Trainable Parameters: {trainable_params:,}")
    summary.append(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    return "\n".join(summary)


def save_model_summary(model: torch.nn.Module, path: str) -> None:
    """
    Lưu bản tóm tắt của mô hình vào file
    
    Args:
        model: Mô hình cần tóm tắt
        path: Đường dẫn để lưu bản tóm tắt
    """
    summary = generate_model_summary(model)
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Lưu bản tóm tắt
    with open(path, 'w') as f:
        f.write(summary)
    
    print(f"Model summary saved to {path}")


def export_model(model: torch.nn.Module, path: str, input_shape: tuple = (1, 1, 32, 140)) -> None:
    """
    Xuất mô hình sang định dạng ONNX
    
    Args:
        model: Mô hình cần xuất
        path: Đường dẫn để lưu mô hình
        input_shape: Kích thước đầu vào của mô hình
    """
    try:
        import onnx
        import onnxruntime
    except ImportError:
        print("Warning: onnx or onnxruntime not installed. Please install them to export models.")
        return
    
    # Đặt mô hình ở chế độ đánh giá
    model.eval()
    
    # Tạo đầu vào giả
    dummy_input = torch.randn(input_shape, requires_grad=True)
    dummy_input = dummy_input.to(next(model.parameters()).device)
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Xuất mô hình
    torch.onnx.export(
        model,                     # Mô hình cần xuất
        dummy_input,               # Đầu vào mẫu
        path,                      # Đường dẫn để lưu mô hình
        export_params=True,        # Lưu các tham số đã huấn luyện
        opset_version=12,          # Phiên bản ONNX
        do_constant_folding=True,  # Gập hằng số
        input_names=['input'],     # Tên của đầu vào
        output_names=['output'],   # Tên của đầu ra
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Kiểm tra mô hình đã xuất
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    
    print(f"Model exported to {path}")