"""
train_optimized.py - Script huấn luyện tối ưu với format gốc (có dấu gạch ngang)
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
    """Thiết lập logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
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


def format_license_plate(text: str) -> str:
    """
    Format license plate với dấu gạch ngang (khôi phục format cũ)
    
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


def decode_predictions_with_format(model, images, config):
    """
    Decode predictions với format có dấu gạch ngang
    
    Args:
        model: VOCR model
        images: Input images
        config: Configuration
        
    Returns:
        List[str]: Decoded predictions với format
    """
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
                # SỬA LỖI: Format với dấu gạch ngang
                prediction = format_license_plate(prediction)
                predictions.append(prediction)
            
            return predictions
    except Exception as e:
        print(f"Error in prediction: {e}")
        return ["" for _ in range(images.size(0))]


def decode_targets_with_format(targets, config):
    """
    Decode targets với format có dấu gạch ngang
    
    Args:
        targets: Target tensor [batch_size, seq_len]
        config: Configuration
        
    Returns:
        List[str]: Decoded target strings với format
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
            # SỬA LỖI: Format với dấu gạch ngang
            target_string = format_license_plate(target_string)
            target_strings.append(target_string)
        
        return target_strings
    except Exception as e:
        print(f"Error decoding targets: {e}")
        return ["" for _ in range(targets.size(0))]


def train_epoch_fast(model, dataloader, criterion, optimizer, scaler, device, config, logger):
    """Training một epoch với tối ưu hóa"""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    # Tham số huấn luyện
    teacher_forcing_ratio = config.get('training.teacher_forcing_ratio', 0.5)
    clip_gradient = config.get('training.clip_gradient', 5.0)
    use_mixed_precision = config.get('training.mixed_precision', True)
    
    # Tăng tốc với tqdm
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, batch_data in enumerate(progress_bar):
        try:
            # Kiểm tra định dạng batch data
            if len(batch_data) == 3:
                images, targets, labels = batch_data
            else:
                images, targets = batch_data
                labels = None
            
            # Chuyển data lên device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
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
                
                loss.backward()
                
                if clip_gradient > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
                
                optimizer.step()
            
            # Cập nhật thống kê
            total_loss += loss.item()
            num_batches += 1
            
            # Cập nhật progress bar mỗi 10 batches để tăng tốc
            if batch_idx % 10 == 0:
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}'
                })
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    return total_loss / max(num_batches, 1)


def validate_epoch_fast(model, dataloader, criterion, device, config, logger, max_batches=50):
    """Validation với format có dấu gạch ngang"""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar):
            # Giới hạn số batches để validation nhanh hơn
            if batch_idx >= max_batches:
                break
                
            try:
                # Kiểm tra định dạng batch data
                if len(batch_data) == 3:
                    images, targets, labels = batch_data
                else:
                    images, targets = batch_data
                    labels = None
                
                # Chuyển data lên device
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Forward pass
                outputs = model(images, targets, teacher_forcing_ratio=0.0)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
                # SỬA LỖI: Decode với format có dấu gạch ngang
                predictions = decode_predictions_with_format(model, images, config)
                
                # Decode targets với format có dấu gạch ngang
                if labels is not None:
                    # SỬA LỖI: Format labels với dấu gạch ngang
                    target_strings = [format_license_plate(str(label)) for label in labels]
                else:
                    target_strings = decode_targets_with_format(targets, config)
                
                # Collect predictions và targets
                min_len = min(len(predictions), len(target_strings))
                if min_len > 0:
                    all_predictions.extend(predictions[:min_len])
                    all_targets.extend(target_strings[:min_len])
                
                # Cập nhật loss
                total_loss += loss.item()
                num_batches += 1
                
                # Quick accuracy estimate
                if len(predictions) > 0 and len(target_strings) > 0:
                    batch_accuracy = sum(p == t for p, t in zip(predictions[:min_len], target_strings[:min_len])) / min_len
                else:
                    batch_accuracy = 0.0
                
                # Update progress bar
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'acc': f'{batch_accuracy:.3f}'
                })
                
            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                continue
    
    # Tính metrics
    if len(all_predictions) > 0 and len(all_targets) > 0:
        try:
            metrics = compute_metrics(all_predictions, all_targets)
            correct_count = sum(1 for p, t in zip(all_predictions, all_targets) if p == t)
            
            # Log sample results
            logger.info(f"Validation: {correct_count}/{len(all_targets)} correct predictions")
            logger.info(f"Sample predictions: {all_predictions[:3]}")
            logger.info(f"Sample targets: {all_targets[:3]}")
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            metrics = {'accuracy': 0.0, 'character_accuracy': 0.0, 'edit_distance': float('inf')}
    else:
        metrics = {'accuracy': 0.0, 'character_accuracy': 0.0, 'edit_distance': float('inf')}
    
    return total_loss / max(num_batches, 1), metrics


def main():
    """Hàm main tối ưu hóa với format cũ"""
    parser = argparse.ArgumentParser(description='Train VOCR Model - Optimized')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    parser.add_argument('--fast', action='store_true', help='Fast training mode')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config()
    
    # Tối ưu hóa config cho training nhanh
    if args.fast:
        config.config['data']['batch_size'] = min(128, config.get('data.batch_size', 64) * 2)
        config.config['training']['epochs'] = min(30, config.get('training.epochs', 100))
        config.config['checkpoint']['save_frequency'] = 10
        print("Fast training mode enabled!")
    
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
        f"{backbone_name}_{timestamp}_optimized"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(save_dir)
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Batch size: {config.get('data.batch_size')}")
    logger.info(f"Total epochs: {config.get('training.epochs')}")
    
    # Save config
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config.config, f, default_flow_style=False)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders()
        logger.info(f"Data loaders created! Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
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
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.get('training.learning_rate', 0.001),
                          weight_decay=float(config.get('training.weight_decay', 1e-5)))
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                    T_max=config.get('training.epochs', 50))
    
    scaler = GradScaler() if config.get('training.mixed_precision', True) else None
    
    # Training parameters
    num_epochs = config.get('training.epochs', 50)
    patience = config.get('training.early_stopping_patience', 10)
    
    # Training loop
    best_val_loss = float('inf')
    best_accuracy = 0.0
    epochs_without_improvement = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        # Training
        train_loss = train_epoch_fast(
            model, train_loader, criterion, optimizer, scaler, device, config, logger
        )
        
        # Validation với giới hạn batches để nhanh hơn
        val_batches_limit = 100 if args.fast else len(val_loader)
        val_loss, metrics = validate_epoch_fast(
            model, val_loader, criterion, device, config, logger, val_batches_limit
        )
        
        # Update learning rate
        scheduler.step()
        
        # Logging
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Character Accuracy: {metrics['character_accuracy']:.4f}")
        logger.info(f"Edit Distance: {metrics['edit_distance']:.4f}")
        
        # Check for improvement
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
            logger.info(f"New best model saved! Accuracy: {best_accuracy:.4f}, Loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % config.get('checkpoint.save_frequency', 10) == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
    
    # Final evaluation
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time/3600:.1f} hours")
    logger.info(f"Best accuracy: {best_accuracy:.4f}, Best validation loss: {best_val_loss:.4f}")
    
    # Quick test set evaluation
    logger.info("\nQuick evaluation on test set...")
    try:
        test_loss, test_metrics = validate_epoch_fast(
            model, test_loader, criterion, device, config, logger, 50
        )
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test Character Accuracy: {test_metrics['character_accuracy']:.4f}")
        logger.info(f"Test Edit Distance: {test_metrics['edit_distance']:.4f}")
    except Exception as e:
        logger.error(f"Error in test evaluation: {str(e)}")


if __name__ == '__main__':
    main()