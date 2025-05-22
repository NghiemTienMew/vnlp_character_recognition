"""
train.py - Script huấn luyện mô hình VOCR (phiên bản sạch)
"""
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import warnings

# Tắt warnings
warnings.filterwarnings("ignore")

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import get_config, load_config
from data.dataset import create_data_loaders
from models.vocr import VOCR
from losses.combined_loss import create_loss_function
from utils.metrics import compute_metrics
from utils.checkpoints import save_checkpoint, load_checkpoint

def decode_targets_to_strings(targets, target_lengths, config):
    """Decode tensor targets back to string labels"""
    idx_to_char = config.get('data.idx_to_char')
    batch_targets = []
    
    for i in range(targets.size(0)):
        target_length = target_lengths[i].item()
        target_indices = targets[i][:target_length].tolist()
        
        # Convert indices to characters, skip special tokens
        chars = []
        for idx in target_indices:
            if idx in idx_to_char and idx not in [0, 1, 2]:  # Skip <pad>, <sos>, <eos>
                chars.append(idx_to_char[idx])
        
        batch_targets.append(''.join(chars))
    
    return batch_targets

def parse_args():
    parser = argparse.ArgumentParser(description='Train VOCR model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    return parser.parse_args()


def setup_optimizer(model, config):
    # Optimizer parameters
    optimizer_name = config.get('training.optimizer', 'adam')
    lr = float(config.get('training.learning_rate', 0.001))
    weight_decay = float(config.get('training.weight_decay', 1e-5))
    
    print(f"Optimizer: {optimizer_name}, lr={lr}, weight_decay={weight_decay}")
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = float(config.get('training.momentum', 0.9))
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Scheduler
    scheduler_name = config.get('training.lr_scheduler', 'cosine')
    if scheduler_name == 'step':
        step_size = int(config.get('training.lr_step_size', 20))
        gamma = float(config.get('training.lr_gamma', 0.1))
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':
        epochs = int(config.get('training.epochs', 100))
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'reduce_on_plateau':
        patience = int(config.get('training.lr_patience', 5))
        factor = float(config.get('training.lr_factor', 0.1))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    model.train()
    total_loss = 0
    
    # Training settings
    teacher_forcing_ratio = float(config.get('training.teacher_forcing_ratio', 0.5))
    clip_gradient = float(config.get('training.clip_gradient', 5.0))
    use_amp = config.get('training.mixed_precision', True) and device.type == 'cuda'
    
    # Setup mixed precision
    scaler = None
    if use_amp:
        try:
            scaler = torch.cuda.amp.GradScaler()
        except:
            use_amp = False
            print("Mixed precision not available, using standard precision")
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, (images, targets, target_lengths) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)  # Thêm dòng này
    
        optimizer.zero_grad()
    
        try:
            if use_amp and scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = model(images, targets, teacher_forcing_ratio)
                    loss = criterion(outputs['outputs'], targets, target_lengths, outputs.get('lm_outputs'))
            
                scaler.scale(loss).backward()
            
                if clip_gradient > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
            
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                outputs = model(images, targets, teacher_forcing_ratio)
                loss = criterion(outputs['outputs'], targets, target_lengths, outputs.get('lm_outputs'))
            
                loss.backward()
            
                if clip_gradient > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
            
                optimizer.step()
        
            batch_loss = loss.item()
            total_loss += batch_loss
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            batch_loss = 0.0
        
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'avg_loss': f'{total_loss / max(1, batch_idx + 1):.4f}'
        })
    
    pbar.close()
    return total_loss / len(train_loader) if len(train_loader) > 0 else 0.0


def validate(model, val_loader, criterion, device, config):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for batch_idx, (images, targets, target_lengths) in enumerate(pbar):  # ← Sửa ở đây
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)  # ← Thêm dòng này
            
            try:
                outputs = model(images, targets, teacher_forcing_ratio=0.0)
                loss = criterion(outputs['outputs'], targets, target_lengths, outputs.get('lm_outputs'))
                
                batch_loss = loss.item()
                total_loss += batch_loss
                
                predictions = model.predict(images)
                all_predictions.extend(predictions)
                
                # Decode targets back to strings for metrics
                # Cần một hàm decode_targets để chuyển targets thành strings
                batch_targets = decode_targets_to_strings(targets, target_lengths, config)
                all_targets.extend(batch_targets)
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                batch_loss = 0.0
            
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{total_loss / max(1, batch_idx + 1):.4f}'
            })
    
    pbar.close()
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    
    if len(all_predictions) > 0 and len(all_targets) > 0:
        metrics = compute_metrics(all_predictions, all_targets)
    else:
        metrics = {'accuracy': 0.0, 'character_accuracy': 0.0, 'edit_distance': 0.0}
    
    return avg_loss, metrics

def decode_targets_to_strings(targets, target_lengths, config):
    """Decode tensor targets back to string labels"""
    idx_to_char = config.get('data.idx_to_char')
    batch_targets = []
    
    for i in range(targets.size(0)):
        target_length = target_lengths[i].item()
        target_indices = targets[i][:target_length].tolist()
        
        # Convert indices to characters, skip special tokens
        chars = []
        for idx in target_indices:
            if idx in idx_to_char and idx not in [0, 1, 2]:  # Skip <pad>, <sos>, <eos>
                chars.append(idx_to_char[idx])
        
        batch_targets.append(''.join(chars))
    
    return batch_targets


def main():
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
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders()
    print("Data loaders created!")
    
    # Create model
    print("Creating model...")
    model = VOCR().to(device)
    print("Model created!")
    
    # Create loss function and optimizer
    print("Setting up training...")
    criterion = create_loss_function()
    optimizer, scheduler = setup_optimizer(model, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume is not None and os.path.isfile(args.resume):
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Create save directory
    save_dir = config.get('checkpoint.save_dir', 'experiments/')
    save_dir = os.path.join(save_dir, f"{config.get('model.backbone')}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config
    config.save(os.path.join(save_dir, 'config.yaml'))
    
    # Training loop
    epochs = int(config.get('training.epochs', 100))
    early_stopping_patience = int(config.get('training.early_stopping_patience', 10))
    no_improvement = 0
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Save directory: {save_dir}")
    
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config)
        
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device, config)
        
        # Update scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Character Accuracy: {metrics['character_accuracy']:.4f}")
        print(f"Edit Distance: {metrics['edit_distance']:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch + 1, val_loss, best_loss,
                os.path.join(save_dir, 'best_model.pth')
            )
            no_improvement = 0
            print(f"New best model saved! Loss: {best_loss:.4f}")
        else:
            no_improvement += 1
        
        # Early stopping
        if no_improvement >= early_stopping_patience:
            print(f"No improvement for {early_stopping_patience} epochs. Early stopping.")
            break
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final evaluation on test set...")
    print("="*60)
    
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model, None)
        test_loss, test_metrics = validate(model, test_loader, criterion, device, config)
        
        print("Test Results:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Character Accuracy: {test_metrics['character_accuracy']:.4f}")
        print(f"Edit Distance: {test_metrics['edit_distance']:.4f}")
        
        # Save test results
        with open(os.path.join(save_dir, 'test_results.txt'), 'w') as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"Character Accuracy: {test_metrics['character_accuracy']:.4f}\n")
            f.write(f"Edit Distance: {test_metrics['edit_distance']:.4f}\n")
    
    print(f"\nTraining completed! Results saved to {save_dir}")


if __name__ == '__main__':
    main()