#!/usr/bin/env python3
"""
Setup script for Football Video Analysis
Automates the installation process for local development
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    if os.path.exists("ml_env"):
        print("⚠️ Virtual environment 'ml_env' already exists")
        response = input("Do you want to recreate it? (y/N): ")
        if response.lower() != 'y':
            return True
        run_command("rmdir /s /q ml_env" if platform.system() == "Windows" else "rm -rf ml_env", "Removing existing environment")
    
    return run_command("python -m venv ml_env", "Creating virtual environment")

def install_pytorch():
    """Install PyTorch with CUDA support"""
    print("🔧 Installing PyTorch with CUDA support...")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass
    
    # Install PyTorch with CUDA
    cuda_commands = [
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    ]
    
    for command in cuda_commands:
        if run_command(command, "Installing PyTorch"):
            # Verify installation
            try:
                import torch
                print(f"✅ PyTorch installed successfully")
                print(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"GPU: {torch.cuda.get_device_name(0)}")
                return True
            except ImportError:
                continue
    
    print("❌ Failed to install PyTorch")
    return False

def install_requirements():
    """Install other requirements"""
    return run_command("pip install -r requirements.txt", "Installing requirements")

def download_models():
    """Download required models"""
    return run_command("python download_reid_model.py", "Downloading ReID model")

def check_model_files():
    """Check if model files exist"""
    required_files = [
        "model/best.pt",
        "osnet_x0_25_market1501.pt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️ Missing model files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("Please ensure these files are present in the project directory")
        return False
    
    print("✅ All model files are present")
    return True

def main():
    """Main setup function"""
    print("⚽ Football Video Analysis Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Activate virtual environment and install packages
    if platform.system() == "Windows":
        activate_cmd = "ml_env\\Scripts\\activate"
    else:
        activate_cmd = "source ml_env/bin/activate"
    
    print(f"🔧 Please activate the virtual environment and run the following commands:")
    print(f"   {activate_cmd}")
    print("   python setup.py --install")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        # Install PyTorch
        if not install_pytorch():
            sys.exit(1)
        
        # Install requirements
        if not install_requirements():
            sys.exit(1)
        
        # Download models
        if not download_models():
            sys.exit(1)
        
        # Check model files
        if not check_model_files():
            sys.exit(1)
        
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 To run the application:")
        print("   streamlit run app.py")
        
    else:
        print("\n📋 Setup script ready!")
        print("Run 'python setup.py --install' after activating the virtual environment")

if __name__ == "__main__":
    main() 