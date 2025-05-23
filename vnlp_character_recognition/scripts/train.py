"""
train.py - Script huấn luyện mô hình VOCR (Final version - đã sửa lỗi Accuracy)
"""
import os
import sys
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import yaml
import logging
from pathlib import Path

# Thêm đường dẫn root vào sys.path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from config.config import get_config, load_config
from data.dataset import create_data_loaders
from models.vocr import VOCR
from losses.combined_loss import create_loss_function
from utils.metrics import compute_metrics
from utils.checkpoints import save_checkpoint, load_checkpoint
from utils.visualize import visualize_predictions, plot_training_curves


def setup_logging(log_dir: str):
    """
    Thiết lập logging
    
    Args:
        log_dir: Thư mục lưu log
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Tạo formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def create_optimizer(model: nn.Module, config):
    """
    Tạo optimizer
    
    Args:
        model: Model cần tối ưu
        config: Cấu hình
        
    Returns:
        torch.optim.Optimizer: Optimizer
    """
    optimizer_type = config.get('training.optimizer', 'adam')
    learning_rate = config.get('training.learning_rate', 0.001)
    weight_decay = float(config.get('training.weight_decay', 1e-5))
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                             weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """
    Tạo learning rate scheduler
    
    Args:
        optimizer: Optimizer
        config: Cấu hình
        
    Returns:
        Learning rate scheduler hoặc None
    """
    scheduler_type = config.get('training.lr_scheduler', 'cosine')
    
    if scheduler_type == 'step':
        step_size = config.get('training.lr_step_size', 20)
        gamma = config.get('training.lr_gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'cosine':
        T_max = config.get('training.epochs', 100)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    
    elif scheduler_type == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
    
    else:
        return None


def decode_predictions(model, images, config):
    """
    Decode predictions từ model outputs
    
    Args:
        model: VOCR model
        images: Input images
        config: Configuration
        
    Returns:
        List[str]: Decoded predictions
    """
    try:
        # Sử dụng model.predict() method
        predictions = model.predict(images)
        return predictions
    except Exception as e:
        print(f"Error in model.predict(): {e}")
        # Fallback: manual decoding
        try:
            with torch.no_grad():
                outputs = model(images, targets=None, teacher_forcing_ratio=0.0)
                logits = outputs['outputs']  # [batch_size, seq_len, num_classes]
                
                # Get predicted indices
                _, predicted_indices = torch.max(logits, dim=2)  # [batch_size, seq_len]
                
                # Decode to strings
                predictions = []
                idx_to_char = config.get('data.idx_to_char', {})
                
                for i in range(predicted_indices.size(0)):
                    indices = predicted_indices[i].cpu().tolist()
                    chars = []
                    
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
                    predictions.append(prediction)
                
                return predictions
        except Exception as e2:
            print(f"Error in manual decoding: {e2}")
            return ["" for _ in range(images.size(0))]


def decode_targets(targets, config):
    """
    Decode targets từ tensor indices
    
    Args:
        targets: Target tensor [batch_size, seq_len]
        config: Configuration
        
    Returns:
        List[str]: Decoded target strings
    """
    try:
        target_strings = []
        idx_to_char = config.get('data.idx_to_char', {})
        
        for i in range(targets.size(0)):
            indices = targets[i].cpu().tolist()
            chars = []
            
            for idx in indices:
                if idx == 0:  # <pad>
                    continue
                elif idx == 1:  # <sos>
                    continue
                elif idx == 2:  # <eos>
                    break
                elif idx in idx_to_char:
                    chars.append(idx_to_char[idx])
            
            target_string = ''.join(chars)
            target_strings.append(target_string)
        
        return target_strings
    except Exception as e:
        print(f"Error decoding targets: {e}")
        return ["" for _ in range(targets.size(0))]


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, config, logger):
    """
    Huấn luyện một epoch
    """
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    # Tham số huấn luyện
    teacher_forcing_ratio = config.get('training.teacher_forcing_ratio', 0.5)
    clip_gradient = config.get('training.clip_gradient', 5.0)
    use_mixed_precision = config.get('training.mixed_precision', True)
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch_data in enumerate(progress_bar):
        try:
            # Kiểm tra định dạng batch data
            if len(batch_data) == 3:
                images, targets, labels = batch_data
            else:
                images, targets = batch_data
                labels = None
            
            # Chuyển data lên device
            images = images.to(device)
            targets = targets.to(device)
            
            # Kiểm tra và xử lý batch size mismatch
            if images.size(0) != targets.size(0):
                min_batch_size = min(images.size(0), targets.size(0))
                images = images[:min_batch_size]
                targets = targets[:min_batch_size]
                logger.warning(f"Batch {batch_idx}: Fixed batch size mismatch")
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass với mixed precision
            if use_mixed_precision:
                with autocast():
                    outputs = model(images, targets, teacher_forcing_ratio)
                    loss_dict = criterion(outputs, targets)
                    loss = loss_dict['total_loss']
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Clip gradients
                if clip_gradient > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images, targets, teacher_forcing_ratio)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                if clip_gradient > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
                
                # Optimizer step
                optimizer.step()
            
            # Cập nhật thống kê
            total_loss += loss.item()
            num_batches += 1
            
            # Cập nhật progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            })
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    return total_loss / max(num_batches, 1)


def validate_epoch(model, dataloader, criterion, device, config, logger):
    """
    Đánh giá một epoch với sửa lỗi Accuracy calculation
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                # Kiểm tra định dạng batch data
                if len(batch_data) == 3:
                    images, targets, labels = batch_data
                else:
                    images, targets = batch_data
                    labels = None
                
                # Chuyển data lên device
                images = images.to(device)
                targets = targets.to(device)
                
                # Kiểm tra và xử lý batch size mismatch
                if images.size(0) != targets.size(0):
                    min_batch_size = min(images.size(0), targets.size(0))
                    images = images[:min_batch_size]
                    targets = targets[:min_batch_size]
                    if labels is not None:
                        labels = labels[:min_batch_size]
                
                # Forward pass
                outputs = model(images, targets, teacher_forcing_ratio=0.0)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
                # SỬA LỖI: Decode predictions và targets đúng cách
                predictions = decode_predictions(model, images, config)
                
                # Decode targets - ưu tiên dùng labels nếu có
                if labels is not None:
                    target_strings = [str(label) for label in labels]
                else:
                    target_strings = decode_targets(targets, config)
                
                # Đảm bảo predictions và targets có cùng length
                min_len = min(len(predictions), len(target_strings))
                if min_len > 0:
                    all_predictions.extend(predictions[:min_len])
                    all_targets.extend(target_strings[:min_len])
                
                # Cập nhật loss
                total_loss += loss.item()
                num_batches += 1
                
                # Cập nhật progress bar với accuracy preview
                if len(all_predictions) > 0 and len(all_targets) > 0:
                    # Tính accuracy nhanh cho một vài samples cuối
                    recent_preds = predictions[:min_len] if len(predictions) > 0 else []
                    recent_targets = target_strings[:min_len] if len(target_strings) > 0 else []
                    
                    if len(recent_preds) > 0 and len(recent_targets) > 0:
                        batch_accuracy = sum(p == t for p, t in zip(recent_preds, recent_targets)) / len(recent_targets)
                    else:
                        batch_accuracy = 0.0
                else:
                    batch_accuracy = 0.0
                
                # Cập nhật progress bar
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'batch_acc': f'{batch_accuracy:.3f}'
                })
                
            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                print(f"Error in validation batch {batch_idx}: {str(e)}")
                continue
    
    # Tính metrics - SỬA LỖI: Đảm bảo có dữ liệu để tính toán
    if len(all_predictions) > 0 and len(all_targets) > 0:
        try:
            # Debug info
            print(f"\nComputing metrics for {len(all_predictions)} predictions and {len(all_targets)} targets")
            print(f"Sample predictions: {all_predictions[:3]}")
            print(f"Sample targets: {all_targets[:3]}")
            
            metrics = compute_metrics(all_predictions, all_targets)
            
            # Log một vài examples để debug
            correct_count = sum(1 for p, t in zip(all_predictions, all_targets) if p == t)
            logger.info(f"Debug: {correct_count}/{len(all_targets)} correct predictions")
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            print(f"Error computing metrics: {str(e)}")
            metrics = {
                'accuracy': 0.0,
                'character_accuracy': 0.0,
                'edit_distance': float('inf')
            }
    else:
        logger.warning("No valid predictions or targets for metric computation")
        metrics = {
            'accuracy': 0.0,
            'character_accuracy': 0.0,
            'edit_distance': float('inf')
        }
    
    return total_loss / max(num_batches, 1), metrics


def main():
    """
    Hàm main cho training
    """
    parser = argparse.ArgumentParser(description='Train VOCR Model')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device to use')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
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
    
    # Create save directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backbone_name = config.get('model.backbone', 'efficientnet_b0')
    save_dir = os.path.join(
        config.get('checkpoint.save_dir', 'experiments'),
        f"{backbone_name}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(save_dir)
    logger.info(f"Save directory: {save_dir}")
    
    # Save config
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config.config, f, default_flow_style=False)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders()
        logger.info("Data loaders created!")
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        return
    
    # Create model
    logger.info("Creating model...")
    try:
        model = VOCR().to(device)
        logger.info("Model created!")
        logger.info(f"Model info: {model.get_model_info()}")
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        return
    
    # Create loss function
    try:
        criterion = create_loss_function()
        logger.info("Loss function created!")
    except Exception as e:
        logger.error(f"Error creating loss function: {str(e)}")
        return
    
    # Setup training
    logger.info("Setting up training...")
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    scaler = GradScaler() if config.get('training.mixed_precision', True) else None
    
    # Training parameters
    num_epochs = config.get('training.epochs', 100)
    patience = config.get('training.early_stopping_patience', 10)
    
    logger.info(f"Optimizer: {config.get('training.optimizer', 'adam')}, "
               f"lr={config.get('training.learning_rate', 0.001)}, "
               f"weight_decay={config.get('training.weight_decay', 1e-5)}")
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    # Training loop
    best_val_loss = float('inf')
    best_accuracy = 0.0
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    accuracies = []
    
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = load_checkpoint(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            logger.info(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        # Training
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config, logger
        )
        
        # Validation
        val_loss, metrics = validate_epoch(
            model, val_loader, criterion, device, config, logger
        )
        
        # Logging
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Character Accuracy: {metrics['character_accuracy']:.4f}")
        logger.info(f"Edit Distance: {metrics['edit_distance']:.4f}")
        
        # Debug info nếu cần
        if args.debug:
            logger.info(f"Debug info:")
            logger.info(f"- Train batches processed: {len(train_loader)}")
            logger.info(f"- Val batches processed: {len(val_loader)}")
            logger.info(f"- Model total parameters: {model.count_parameters()}")
        
        # Update learning rate
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(metrics['accuracy'])
        
        # Check for improvement - SỬA LỖI: Ưu tiên accuracy hơn loss
        current_accuracy = metrics['accuracy']
        is_best = (current_accuracy > best_accuracy) or (current_accuracy == best_accuracy and val_loss < best_val_loss)
        
        if is_best:
            best_val_loss = val_loss
            best_accuracy = current_accuracy
            epochs_without_improvement = 0
            
            # Save best model
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(save_dir, 'best_model.pth')
            )
            logger.info(f"Checkpoint saved to {save_dir}/best_model.pth")
            logger.info(f"New best model saved! Accuracy: {best_accuracy:.4f}, Loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % config.get('checkpoint.save_frequency', 5) == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    try:
        test_loss, test_metrics = validate_epoch(
            model, test_loader, criterion, device, config, logger
        )
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test Character Accuracy: {test_metrics['character_accuracy']:.4f}")
        logger.info(f"Test Edit Distance: {test_metrics['edit_distance']:.4f}")
    except Exception as e:
        logger.error(f"Error in test evaluation: {str(e)}")
    
    # Plot training curves
    try:
        plot_training_curves(
            train_losses, val_losses, accuracies,
            save_path=os.path.join(save_dir, 'training_curves.png')
        )
    except Exception as e:
        logger.error(f"Error plotting training curves: {str(e)}")
    
    # Save final predictions sample
    try:
        model.eval()
        with torch.no_grad():
            # Get a batch from validation set
            val_iter = iter(val_loader)
            batch_data = next(val_iter)
            
            if len(batch_data) == 3:
                images, targets, labels = batch_data
            else:
                images, targets = batch_data
                labels = None
            
            images = images.to(device)
            predictions = decode_predictions(model, images, config)
            target_strings = [str(label) for label in labels] if labels else decode_targets(targets, config)
            
            # Save sample predictions
            with open(os.path.join(save_dir, 'sample_predictions.txt'), 'w', encoding='utf-8') as f:
                f.write("Sample Predictions vs Targets:\n")
                f.write("="*50 + "\n")
                for i, (pred, target) in enumerate(zip(predictions[:10], target_strings[:10])):
                    f.write(f"{i+1:2d}. Pred: '{pred}' | Target: '{target}' | Match: {pred == target}\n")
            
    except Exception as e:
        logger.error(f"Error saving sample predictions: {str(e)}")
    
    logger.info(f"Training completed! Best accuracy: {best_accuracy:.4f}, "
               f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()