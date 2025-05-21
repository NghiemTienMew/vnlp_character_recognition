"""
train.py - Script huấn luyện mô hình VOCR
"""
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
from utils.visualize import visualize_attention, visualize_predictions


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train VOCR model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()


def setup_optimizer(model, config):
    """
    Setup optimizer and learning rate scheduler
    
    Args:
        model: Model to optimize
        config: Configuration object
        
    Returns:
        Tuple: (optimizer, scheduler)
    """
    # Optimizer
    optimizer_name = config.get('training.optimizer', 'adam')
    lr = config.get('training.learning_rate', 0.001)
    weight_decay = config.get('training.weight_decay', 1e-5)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('training.momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Scheduler
    scheduler_name = config.get('training.lr_scheduler', 'cosine')
    
    if scheduler_name == 'step':
        step_size = config.get('training.lr_step_size', 20)
        gamma = config.get('training.lr_gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':
        epochs = config.get('training.epochs', 100)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'reduce_on_plateau':
        patience = config.get('training.lr_patience', 5)
        factor = config.get('training.lr_factor', 0.1)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch
        config: Configuration object
        
    Returns:
        float: Average loss for this epoch
    """
    model.train()
    
    total_loss = 0
    correct_chars = 0
    total_chars = 0
    batch_loss = 0
    
    # Training mode settings
    teacher_forcing_ratio = config.get('training.teacher_forcing_ratio', 0.5)
    clip_gradient = config.get('training.clip_gradient', 5.0)
    
    # Tqdm progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    # Mixed precision training if enabled
    use_amp = config.get('training.mixed_precision', True)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for batch_idx, (images, targets, original_labels) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        print("target shape:", targets.shape)
        print("target max index:", targets.max().item())
        print("num_classes:", config.get("data.num_classes"))
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images, targets, teacher_forcing_ratio)
                loss = criterion(outputs['outputs'], targets, outputs['lm_outputs'])
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Clip gradients
            if clip_gradient > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
            
            # Update weights with gradient scaling
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward and backward pass
            outputs = model(images, targets, teacher_forcing_ratio)
            loss = criterion(outputs['outputs'], targets, outputs['lm_outputs'])
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            if clip_gradient > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
            
            # Update weights
            optimizer.step()
        
        # Calculate metrics
        batch_loss = loss.item()
        total_loss += batch_loss
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    # Close progress bar
    pbar.close()
    
    # Calculate average loss
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss


def validate(model, val_loader, criterion, device, config):
    """
    Validate the model
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        config: Configuration object
        
    Returns:
        Tuple[float, Dict]: (average loss, metrics dictionary)
    """
    model.eval()
    
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    # Tqdm progress bar
    pbar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for batch_idx, (images, targets, original_labels) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images, targets, teacher_forcing_ratio=0.0)
            loss = criterion(outputs['outputs'], targets, None)
            
            # Calculate metrics
            batch_loss = loss.item()
            total_loss += batch_loss
            
            # Get predictions
            predictions = model.predict(images)
            
            # Store predictions and targets for metrics calculation
            all_predictions.extend(predictions)
            all_targets.extend(original_labels)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
    
    # Close progress bar
    pbar.close()
    
    # Calculate average loss
    avg_loss = total_loss / len(val_loader)
    
    # Calculate metrics
    metrics = compute_metrics(all_predictions, all_targets)
    
    return avg_loss, metrics


def visualize_results(model, val_loader, device, save_dir, epoch):
    """
    Visualize prediction results
    
    Args:
        model: Model to use
        val_loader: Validation data loader
        device: Device to use
        save_dir: Directory to save visualizations
        epoch: Current epoch
    """
    model.eval()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get a batch of data
    images, targets, original_labels = next(iter(val_loader))
    images = images.to(device)
    targets = targets.to(device)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(images, targets, teacher_forcing_ratio=0.0)
        
        # Get predictions
        predictions = model.predict(images)
        
        # Visualize predictions
        fig = visualize_predictions(images, predictions, original_labels)
        plt.savefig(os.path.join(save_dir, f'predictions_epoch_{epoch+1}.png'))
        plt.close(fig)
        
        # Visualize attention if available
        if outputs['attention_weights'] is not None:
            attention_weights = outputs['attention_weights'].cpu().numpy()
            for i in range(min(4, images.size(0))):
                fig = visualize_attention(
                    images[i], 
                    attention_weights[i], 
                    predictions[i], 
                    original_labels[i]
                )
                plt.savefig(os.path.join(save_dir, f'attention_epoch_{epoch+1}_sample_{i+1}.png'))
                plt.close(fig)


def main():
    """
    Main training function
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
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Create model
    model = VOCR().to(device)
    
    # Create loss function
    criterion = create_loss_function()
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer(model, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = load_checkpoint(args.resume, model, optimizer)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            print(f"Resumed from checkpoint: {args.resume}")
            print(f"Starting from epoch {start_epoch+1}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Create save directory
    save_dir = config.get('checkpoint.save_dir', 'experiments/')
    save_dir = os.path.join(save_dir, f"{config.get('model.backbone')}_{time.strftime('%Y%m%d_%H%M%S')}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save config
    config.save(os.path.join(save_dir, 'config.yaml'))
    
    # Training loop
    epochs = config.get('training.epochs', 100)
    save_frequency = config.get('checkpoint.save_frequency', 5)
    save_best_only = config.get('checkpoint.save_best_only', True)
    early_stopping_patience = config.get('training.early_stopping_patience', 10)
    
    # For early stopping
    no_improvement = 0
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Save directory: {save_dir}")
    
    for epoch in range(start_epoch, epochs):
        # Train one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config)
        
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device, config)
        
        # Update scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Print metrics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}, Character Accuracy: {metrics['character_accuracy']:.4f}")
        print(f"Edit Distance: {metrics['edit_distance']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_frequency == 0 and not save_best_only:
            save_checkpoint(
                model, optimizer, epoch + 1, val_loss, best_loss,
                os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch + 1, val_loss, best_loss,
                os.path.join(save_dir, 'best_model.pth')
            )
            no_improvement = 0
            
            # Visualize results for best model
            visualize_dir = os.path.join(save_dir, 'visualizations')
            visualize_results(model, val_loader, device, visualize_dir, epoch)
        else:
            no_improvement += 1
        
        # Early stopping
        if no_improvement >= early_stopping_patience:
            print(f"No improvement for {early_stopping_patience} epochs. Early stopping.")
            break
    
    # Evaluate on test set
    print("Evaluating best model on test set...")
    
    # Load best model
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    load_checkpoint(best_model_path, model, None)
    
    # Evaluate
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
        
        if 'confusion_matrix' in test_metrics:
            f.write("\nConfusion Matrix:\n")
            f.write(str(test_metrics['confusion_matrix']))
    
    print(f"Training completed. Results saved to {save_dir}")


if __name__ == '__main__':
    main()