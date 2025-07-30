#!/usr/bin/env python3
"""
Test script for the fallback system
Verifies that the app works with both GPU and CPU configurations
"""

import torch
import psutil
import sys
from pathlib import Path

def test_system_capabilities():
    """Test and display system capabilities"""
    print("🔍 Testing System Capabilities")
    print("=" * 40)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check PyTorch and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {gpu_memory:.1f} GB")
    else:
        print("⚠️ CUDA not available - will use CPU fallback")
    
    # Check CPU and memory
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"CPU cores: {cpu_count}")
    print(f"RAM: {memory_gb:.1f} GB")
    
    # Determine optimal configuration
    if torch.cuda.is_available():
        config = {
            'device': 'cuda',
            'batch_size': 16,
            'half_precision': True,
            'optimization_level': 'high'
        }
        
        # Adjust batch size based on GPU memory
        if gpu_memory < 4:
            config['batch_size'] = 8
        elif gpu_memory > 8:
            config['batch_size'] = 32
            
        print(f"✅ Recommended: GPU configuration")
    else:
        config = {
            'device': 'cpu',
            'batch_size': 4,
            'half_precision': False,
            'optimization_level': 'balanced'
        }
        
        # Adjust based on CPU capabilities
        if cpu_count >= 8 and memory_gb >= 16:
            config['batch_size'] = 8
            config['optimization_level'] = 'high'
        elif cpu_count < 4 or memory_gb < 8:
            config['batch_size'] = 2
            config['optimization_level'] = 'conservative'
            
        print(f"✅ Recommended: CPU configuration")
    
    print(f"Configuration: {config}")
    return config

def test_model_files():
    """Test if required model files exist"""
    print("\n📁 Testing Model Files")
    print("=" * 40)
    
    required_files = [
        "model/best.pt",
        "osnet_x0_25_market1501.pt"
    ]
    
    all_exist = True
    for file_path in required_files:
        exists = Path(file_path).exists()
        status = "✅" if exists else "❌"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("✅ All model files are present")
    else:
        print("⚠️ Some model files are missing")
    
    return all_exist

def test_tracker_initialization(config):
    """Test tracker initialization with the given configuration"""
    print(f"\n🤖 Testing Tracker Initialization ({config['device'].upper()})")
    print("=" * 40)
    
    try:
        from tracker import Tracker
        
        # Test initialization
        tracker = Tracker("model/best.pt", processing_config=config)
        print("✅ Tracker initialized successfully")
        
        # Test device assignment
        expected_device = torch.device(config['device'])
        if tracker.device == expected_device:
            print(f"✅ Device correctly assigned: {tracker.device}")
        else:
            print(f"⚠️ Device mismatch: expected {expected_device}, got {tracker.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tracker initialization failed: {e}")
        return False

def main():
    """Main test function"""
    print("⚽ Football Video Analysis - Fallback System Test")
    print("=" * 50)
    
    # Test system capabilities
    config = test_system_capabilities()
    
    # Test model files
    models_ok = test_model_files()
    
    # Test tracker initialization
    if models_ok:
        tracker_ok = test_tracker_initialization(config)
    else:
        print("⚠️ Skipping tracker test due to missing model files")
        tracker_ok = False
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 40)
    print(f"System configuration: {config['device'].upper()}")
    print(f"Model files: {'✅ OK' if models_ok else '❌ Missing'}")
    print(f"Tracker initialization: {'✅ OK' if tracker_ok else '❌ Failed'}")
    
    if models_ok and tracker_ok:
        print("\n🎉 All tests passed! The fallback system is working correctly.")
        print(f"💡 The app will run with {config['device'].upper()} optimization.")
    else:
        print("\n⚠️ Some tests failed. Please check the issues above.")
    
    return models_ok and tracker_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 