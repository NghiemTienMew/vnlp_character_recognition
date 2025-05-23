"""
predict.py - Script dự đoán cho mô hình VOCR
"""
import os
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import time

from config.config import get_config, load_config
from models.vocr import VOCR
from data.preprocessor import Preprocessor
from utils.checkpoints import load_checkpoint
from utils.visualize import visualize_predictions, visualize_attention


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Predict with VOCR model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Directory to save results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for benchmark')
    
    return parser.parse_args()


def preprocess_image(image_path, preprocessor):
    """
    Tiền xử lý ảnh
    
    Args:
        image_path: Đường dẫn đến ảnh
        preprocessor: Bộ tiền xử lý
        
    Returns:
        torch.Tensor: Tensor đã tiền xử lý
    """
    # Đọc ảnh
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Tiền xử lý
    tensor = preprocessor(image)
    
    return tensor.unsqueeze(0)  # Thêm chiều batch


def predict_single(model, image_path, device, preprocessor):
    """
    Dự đoán cho một ảnh
    
    Args:
        model: Mô hình
        image_path: Đường dẫn đến ảnh
        device: Device để chạy model
        preprocessor: Bộ tiền xử lý
        
    Returns:
        Tuple: (Dự đoán, Thời gian xử lý, Attention weights)
    """
    # Tiền xử lý ảnh
    tensor = preprocess_image(image_path, preprocessor)
    tensor = tensor.to(device)
    
    # Chuyển model sang eval mode
    model.eval()
    
    # Dự đoán
    with torch.no_grad():
        # Đo thời gian
        start_time = time.time()
        
        # Forward pass
        outputs = model(tensor)
        predictions = model.predict(tensor)
        
        # Kết thúc đo thời gian
        end_time = time.time()
        inference_time = end_time - start_time
    
    # Lấy dự đoán đầu tiên (vì batch_size=1)
    prediction = predictions[0]
    
    # Lấy attention weights nếu có
    attention_weights = None
    if 'attention_weights' in outputs and outputs['attention_weights'] is not None:
        attention_weights = outputs['attention_weights'][0].cpu().numpy()
    
    return prediction, inference_time, attention_weights


def predict_batch(model, image_paths, device, preprocessor, batch_size=16):
    """
    Dự đoán cho nhiều ảnh
    
    Args:
        model: Mô hình
        image_paths: Danh sách đường dẫn đến ảnh
        device: Device để chạy model
        preprocessor: Bộ tiền xử lý
        batch_size: Kích thước batch
        
    Returns:
        Tuple: (Danh sách dự đoán, Thời gian xử lý trung bình)
    """
    # Chuyển model sang eval mode
    model.eval()
    
    # Danh sách kết quả
    predictions = []
    inference_times = []
    
    # Tqdm progress bar
    pbar = tqdm(range(0, len(image_paths), batch_size), desc='Predicting')
    
    for i in pbar:
        # Lấy batch
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors = []
        
        # Tiền xử lý từng ảnh
        for path in batch_paths:
            try:
                tensor = preprocess_image(path, preprocessor)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                # Thêm tensor rỗng để giữ đúng thứ tự
                batch_tensors.append(torch.zeros((1, 1, 32, 140)))
        
        # Ghép các tensor thành batch
        batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
        
        # Dự đoán
        with torch.no_grad():
            # Đo thời gian
            start_time = time.time()
            
            # Forward pass
            batch_predictions = model.predict(batch_tensor)
            
            # Kết thúc đo thời gian
            end_time = time.time()
            batch_time = end_time - start_time
            
            # Tính thời gian trung bình cho mỗi ảnh
            avg_time = batch_time / len(batch_paths)
            inference_times.append(avg_time)
        
        # Lưu các dự đoán
        predictions.extend(batch_predictions)
        
        # Cập nhật progress bar
        pbar.set_postfix({'avg_time': f'{avg_time*1000:.2f} ms'})
    
    # Tính thời gian trung bình
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    return predictions, avg_inference_time


def benchmark(model, device, input_shape=(1, 1, 32, 140), num_runs=100, batch_size=1):
    """
    Benchmark mô hình
    
    Args:
        model: Mô hình
        device: Device để chạy model
        input_shape: Kích thước đầu vào
        num_runs: Số lần chạy
        batch_size: Kích thước batch
        
    Returns:
        Dict: Kết quả benchmark
    """
    # Chuyển model sang eval mode
    model.eval()
    
    # Tạo dữ liệu đầu vào ngẫu nhiên
    input_tensor = torch.randn((batch_size,) + input_shape[1:]).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Benchmark
    inference_times = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc='Benchmarking'):
            # Đo thời gian
            start_time = time.time()
            
            # Forward pass
            _ = model(input_tensor)
            
            # Đồng bộ hóa GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Kết thúc đo thời gian
            end_time = time.time()
            inference_time = end_time - start_time
            
            inference_times.append(inference_time)
    
    # Tính các thống kê
    avg_time = sum(inference_times) / len(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    
    # Tính FPS
    fps = batch_size / avg_time
    
    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'fps': fps,
        'batch_size': batch_size
    }


def main():
    """
    Main prediction function
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
    
    # Create model
    model = VOCR().to(device)
    
    # Load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise ValueError(f"No checkpoint found at {args.checkpoint}")
    
    checkpoint = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from {args.checkpoint} (epoch {checkpoint['epoch']})")
    
    # Create preprocessor
    preprocessor = Preprocessor()
    
    # Create save directory
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Run benchmark if requested
    if args.benchmark:
        print("Running benchmark...")
        benchmark_results = benchmark(
            model, device, input_shape=(1, 1, 32, 140), 
            num_runs=100, batch_size=args.batch_size
        )
        
        print(f"Benchmark Results:")
        print(f"Average Inference Time: {benchmark_results['avg_time']*1000:.2f} ms")
        print(f"Min Inference Time: {benchmark_results['min_time']*1000:.2f} ms")
        print(f"Max Inference Time: {benchmark_results['max_time']*1000:.2f} ms")
        print(f"FPS: {benchmark_results['fps']:.2f}")
        print(f"Batch Size: {benchmark_results['batch_size']}")
        
        # Lưu kết quả benchmark
        with open(os.path.join(save_dir, 'benchmark_results.txt'), 'w') as f:
            for key, value in benchmark_results.items():
                if key.endswith('time'):
                    f.write(f"{key}: {value*1000:.2f} ms\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        return
    
    # Kiểm tra đường dẫn đầu vào
    if os.path.isfile(args.input):
        # Dự đoán cho một ảnh
        prediction, inference_time, attention_weights = predict_single(
            model, args.input, device, preprocessor
        )
        
        print(f"Prediction: {prediction}")
        print(f"Inference Time: {inference_time*1000:.2f} ms")
        
        # Lưu kết quả
        with open(os.path.join(save_dir, 'predictions.txt'), 'w') as f:
            f.write(f"Image: {os.path.basename(args.input)}\n")
            f.write(f"Prediction: {prediction}\n")
            f.write(f"Inference Time: {inference_time*1000:.2f} ms\n")
        
        # Trực quan hóa nếu cần
        if args.visualize:
            # Đọc ảnh
            image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
            
            # Tiền xử lý
            tensor = preprocessor(image)
            
            # Trực quan hóa dự đoán
            fig = visualize_predictions(
                tensor.unsqueeze(0), [prediction], [os.path.basename(args.input)]
            )
            plt.savefig(os.path.join(save_dir, 'prediction.png'))
            plt.close(fig)
            
            # Trực quan hóa attention nếu có
            if attention_weights is not None:
                fig = visualize_attention(
                    tensor, attention_weights, prediction, os.path.basename(args.input)
                )
                plt.savefig(os.path.join(save_dir, 'attention.png'))
                plt.close(fig)
    
    elif os.path.isdir(args.input):
        # Dự đoán cho một thư mục
        image_paths = []
        
        # Tìm tất cả các file ảnh trong thư mục
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            print(f"No images found in {args.input}")
            return
        
        print(f"Found {len(image_paths)} images")
        
        # Dự đoán
        predictions, avg_inference_time = predict_batch(
            model, image_paths, device, preprocessor, batch_size=args.batch_size
        )
        
        print(f"Average Inference Time: {avg_inference_time*1000:.2f} ms")
        print(f"FPS: {1/avg_inference_time:.2f}")
        
        # Lưu kết quả
        with open(os.path.join(save_dir, 'predictions.txt'), 'w') as f:
            for path, prediction in zip(image_paths, predictions):
                f.write(f"{os.path.relpath(path, args.input)},{prediction}\n")
            
            f.write(f"\nAverage Inference Time: {avg_inference_time*1000:.2f} ms\n")
            f.write(f"FPS: {1/avg_inference_time:.2f}\n")
        
        # Trực quan hóa nếu cần
        if args.visualize:
            # Chọn một số mẫu để trực quan hóa
            num_samples = min(10, len(image_paths))
            sample_indices = np.linspace(0, len(image_paths) - 1, num_samples, dtype=int)
            
            sample_images = []
            sample_predictions = []
            sample_filenames = []
            
            for idx in sample_indices:
                path = image_paths[idx]
                prediction = predictions[idx]
                
                try:
                    # Đọc và tiền xử lý ảnh
                    tensor = preprocess_image(path, preprocessor)
                    sample_images.append(tensor.squeeze(0))
                    sample_predictions.append(prediction)
                    sample_filenames.append(os.path.basename(path))
                except Exception as e:
                    print(f"Error processing {path}: {e}")
            
            if sample_images:
                # Ghép các tensor
                sample_tensor = torch.stack(sample_images)
                
                # Trực quan hóa dự đoán
                fig = visualize_predictions(
                    sample_tensor, sample_predictions, sample_filenames
                )
                plt.savefig(os.path.join(save_dir, 'sample_predictions.png'))
                plt.close(fig)
    
    else:
        print(f"Input path {args.input} does not exist")


if __name__ == '__main__':
    main()