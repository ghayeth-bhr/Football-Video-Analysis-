#!/usr/bin/env python3
"""
Script to download the ReID model required for StrongSort
"""

import os
import gdown
from pathlib import Path

def download_reid_model():
    """Download OSNet ReID model if it doesn't exist"""
    
    model_path = Path('osnet_x0_25_market1501.pt')
    
    if model_path.exists():
        print(f"✅ ReID model already exists at: {model_path}")
        return True
    
    print("📥 Downloading OSNet ReID model...")
    
    # OSNet model URL (lightweight version for GPU)
    url = "https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJehkgq_M"
    
    try:
        # Download the model
        gdown.download(url, str(model_path), quiet=False)
        
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ ReID model downloaded successfully!")
            print(f"   File: {model_path}")
            print(f"   Size: {file_size:.1f} MB")
            return True
        else:
            print("❌ Failed to download ReID model")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading ReID model: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Downloading ReID model for StrongSort")
    print("=" * 40)
    
    success = download_reid_model()
    
    if success:
        print("\n🎉 ReID model ready for use!")
        print("   StrongSort can now use GPU with full acceleration.")
    else:
        print("\n❌ Failed to download ReID model.")
        print("   Check your internet connection and try again.")

if __name__ == "__main__":
    main() 