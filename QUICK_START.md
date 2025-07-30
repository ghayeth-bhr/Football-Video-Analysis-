# ⚽ Quick Start Guide

Get your Football Video Analysis app running in minutes!

## 🚀 Fast Setup (5 minutes)

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd Football-Video-Analysis
python setup.py
```

### 2. Activate Environment & Install
```bash
# Windows:
ml_env\Scripts\activate

# macOS/Linux:
source ml_env/bin/activate

# Install everything:
python setup.py --install
```

### 3. Run the App
```bash
streamlit run app.py
```

🎉 **That's it!** Your app is running at `http://localhost:8501`

---

## 📱 How to Use

### Step 1: Upload Video
- Click "Browse files" in the sidebar
- Select your football video (MP4, AVI, MOV, MKV)
- Recommended: 720p or 1080p, under 200MB

### Step 2: Check Settings
- Verify GPU is detected (green checkmark)
- Ensure models are loaded
- Choose processing options

### Step 3: Process
- Click "🚀 Process Video"
- Wait for analysis (2-10 minutes depending on video length)

### Step 4: View Results
- **Team Statistics**: Ball control and possession
- **Player Performance**: Speed and distance metrics
- **Processed Video**: Download annotated video
- **Detailed Analysis**: Key insights and highlights

---

## 🔧 Troubleshooting

### Test Fallback System
```bash
# Test if the fallback system works correctly
python test_fallback.py
```

### GPU Not Working?
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CPU Fallback Issues?
```bash
# The app automatically falls back to CPU if GPU is not available
# Check CPU optimization settings in the app sidebar
# Adjust frame skip and resolution scale for better performance
```

### Models Missing?
```bash
# Download ReID model
python download_reid_model.py

# Check if files exist
ls model/best.pt
ls osnet_x0_25_market1501.pt
```

### OpenCV Issues?
```bash
pip uninstall opencv-python
pip install opencv-python
```

---

## ⚡ Performance Tips

### For Speed:
- ✅ Use GPU acceleration (automatic if available)
- ✅ Use 720p videos instead of 1080p
- ✅ Close other applications
- ✅ On CPU: Use frame skip and resolution scaling

### For Accuracy:
- ✅ Use 1080p videos
- ✅ Ensure good video quality and lighting
- ✅ On CPU: Disable frame skip for full frame processing

---

## 📊 Expected Times

| Video Length | Resolution | GPU | CPU |
|-------------|------------|-----|-----|
| 30 seconds  | 720p       | 2-3 min | 10-15 min |
| 1 minute    | 720p       | 5-7 min | 25-35 min |
| 2 minutes   | 720p       | 10-15 min | 50-70 min |

---

## 🆘 Need Help?

1. Check the full [README.md](README.md) for detailed instructions
2. Verify your GPU and CUDA installation
3. Ensure all model files are present
4. Check video format and quality

---

**⚽ Happy analyzing!** 