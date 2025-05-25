"""
fix_checkpoint.py - Extract và fix checkpoint loading
"""
import torch
import os


def fix_checkpoint(input_path, output_path):
    """
    Fix checkpoint để có thể load được với PyTorch 2.6
    
    Args:
        input_path: Đường dẫn checkpoint gốc
        output_path: Đường dẫn checkpoint đã fix
    """
    try:
        print(f"📂 Loading checkpoint: {input_path}")
        
        # Load với weights_only=False để bypass restriction
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        
        print(f"🔍 Checkpoint contains: {list(checkpoint.keys())}")
        
        # Extract chỉ model state dict
        if 'model_state_dict' in checkpoint:
            model_weights = checkpoint['model_state_dict']
            
            # In thông tin về model
            print(f"📊 Model has {len(model_weights)} parameters")
            print(f"📋 Sample keys: {list(model_weights.keys())[:5]}")
            
            # Lưu chỉ model weights
            torch.save(model_weights, output_path)
            print(f"✅ Fixed checkpoint saved to: {output_path}")
            
            # Verify
            test_load = torch.load(output_path, map_location='cpu')
            print(f"✅ Verification successful: {len(test_load)} parameters")
            
            return True
        else:
            print("❌ No model_state_dict found!")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_checkpoint_info(checkpoint_path):
    """
    In thông tin chi tiết về checkpoint
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print("="*60)
        print("CHECKPOINT INFORMATION")
        print("="*60)
        
        if isinstance(checkpoint, dict):
            for key, value in checkpoint.items():
                if key == 'model_state_dict':
                    print(f"📊 {key}: {len(value)} parameters")
                elif key in ['epoch', 'best_val_loss', 'val_loss']:
                    print(f"📈 {key}: {value}")
                else:
                    print(f"🔧 {key}: {type(value)}")
        else:
            print(f"⚠️  Checkpoint is not a dict: {type(checkpoint)}")
            
    except Exception as e:
        print(f"❌ Error reading checkpoint: {e}")


if __name__ == '__main__':
    # Paths
    input_checkpoint = "/kaggle/working/vnlp_character_recognition/experiments/best_model.pth"
    output_checkpoint = "/kaggle/working/vnlp_character_recognition/experiments/model_weights_fixed.pth"
    
    print("🔧 CHECKPOINT FIXER")
    print("="*50)
    
    # Kiểm tra file tồn tại
    if not os.path.exists(input_checkpoint):
        print(f"❌ Checkpoint not found: {input_checkpoint}")
        exit(1)
    
    # In thông tin checkpoint
    print("📋 Analyzing original checkpoint...")
    test_checkpoint_info(input_checkpoint)
    
    # Fix checkpoint
    print("\n🔧 Fixing checkpoint...")
    success = fix_checkpoint(input_checkpoint, output_checkpoint)
    
    if success:
        print(f"\n✅ SUCCESS!")
        print(f"📝 Use this command to test:")
        print(f"!python test.py --image /kaggle/working/vnlp_character_recognition/one_row_36A02391.png --checkpoint {output_checkpoint}")
    else:
        print(f"\n❌ FAILED to fix checkpoint!")