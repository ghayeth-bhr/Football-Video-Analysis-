# ⚽ Football Video Analysis

A comprehensive football video analysis application with GPU-optimized object tracking, player performance analysis, and team statistics.

## 🛠️ Technologies Used

### **🤖 Machine Learning & AI**
- **PyTorch 2.0+**: Deep learning framework with CUDA acceleration
- **YOLO (You Only Look Once)**: Real-time object detection for players, ball, and referees
- **StrongSORT**: Advanced multi-object tracking algorithm
- **ReID (Re-Identification)**: Player identification and tracking consistency
- **Ultralytics**: YOLO model management and inference

### **🎥 Computer Vision**
- **OpenCV**: Video processing, frame extraction, and image manipulation
- **NumPy**: Numerical computing and array operations
- **PIL (Pillow)**: Image processing and manipulation

### **🌐 Web Interface**
- **Streamlit**: Interactive web application framework
- **Plotly**: Interactive data visualizations and charts
- **Pandas**: Data manipulation and analysis

### **⚡ Performance & Optimization**
- **CUDA**: GPU acceleration for deep learning models
- **Half Precision (FP16)**: Memory optimization for GPU processing
- **Batch Processing**: Optimized inference pipeline
- **psutil**: System monitoring and resource management

### **🔧 Development & Deployment**
- **Python 3.8+**: Core programming language
- **Git**: Version control
- **Virtual Environments**: Dependency isolation
- **Requirements Management**: Automated dependency installation



### **📋 Version Compatibility**
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Core runtime |
| PyTorch | 2.0+ | Deep learning |
| CUDA | 11.0+ | GPU acceleration |
| Streamlit | 1.28+ | Web interface |
| OpenCV | 4.8+ | Computer vision |
| YOLO | 8.0+ | Object detection |
| StrongSORT | 15.0+ | Object tracking |

## 🚀 Features

- **GPU-Optimized Tracking**: Maximum GPU utilization with CUDA acceleration
- **Player Detection**: YOLO-based player, referee, and ball detection
- **Object Tracking**: StrongSORT algorithm for consistent player tracking
- **Performance Analysis**: Player statistics, ball control, and team metrics
- **Web Interface**: Streamlit-based user-friendly interface
- **Real-time Processing**: Fast video analysis with GPU acceleration

## 🎥 Demo Images

Here are some screenshots showcasing the application's capabilities:

### 📊 Team Statistics Dashboard
![Team Statistics](asset/Screenshot%202025-07-30%20192719.png)

### 👥 Player Performance Analysis
![Player Performance](asset/Screenshot%202025-07-30%20192652.png)

### 🎥 Video Processing Interface
![Video Processing](asset/Screenshot%202025-07-30%20192546.png)

### 📈 Detailed Analysis Results
![Detailed Analysis](asset/Screenshot%202025-07-30%20192532.png)

### 🔧 Application Settings
![Application Settings](asset/Screenshot%202025-07-30%20192511.png)

*These screenshots demonstrate the real-time player tracking, ball detection, and comprehensive performance analysis features of the application.*

## 🛠️ Local Installation Guide

### Prerequisites
- **Python 3.8+**
- **NVIDIA GPU with CUDA support** (recommended for best performance)
- **CUDA Toolkit 11.0+** and **cuDNN**
- **Git**

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd Football-Video-Analysis
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv ml_env

# Activate virtual environment
# On Windows:
ml_env\Scripts\activate
# On macOS/Linux:
source ml_env/bin/activate
```

### Step 3: Install PyTorch with CUDA Support
```bash
# Install PyTorch with CUDA (adjust CUDA version as needed)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### Step 4: Install Other Dependencies
```bash
# Install remaining requirements
pip install -r requirements.txt
```

### Step 5: Download Required Models
```bash
# Download ReID model (if not already present)
python download_reid_model.py
```

### Step 6: Verify Model Files
Ensure you have these files in your project:
- `model/best.pt` - YOLO detection model
- `osnet_x0_25_market1501.pt` - ReID model for tracking

## 🎯 Running the Application

### Method 1: Using Streamlit (Recommended)
```bash
# Make sure your virtual environment is activated
streamlit run app.py
```

### Method 2: Direct Python Execution
```bash
# Run the main analysis script directly
python main.py
```

The Streamlit app will open in your browser at `http://localhost:8501`

## 📱 How to Use the App

### 1. **Upload Video**
- Use the sidebar to upload a football video file
- Supported formats: MP4, AVI, MOV, MKV
- Recommended video quality: 720p or 1080p
- Maximum file size: 200MB (adjustable in config)

### 2. **Configure Settings**
- **GPU Status**: Check if GPU acceleration is available
- **Model Selection**: Verify YOLO and ReID models are loaded
- **Processing Options**: Choose between full processing or stub files

### 3. **Process Video**
- Click "🚀 Process Video" to start analysis
- Processing time depends on video length and GPU performance
- Progress will be shown with a spinner

### 4. **View Results**
The app provides results in four tabs:

#### 📊 **Team Statistics**
- Ball control distribution (pie chart)
- Team performance comparison (bar chart)
- Possession percentages

#### 👥 **Player Performance**
- Individual player speed analysis
- Distance covered statistics
- Player performance rankings

#### 🎥 **Processed Video**
- Sample frames preview
- Download processed video with annotations
- Video information and statistics

#### 📈 **Detailed Analysis**
- Summary metrics
- Key insights and highlights
- Performance recommendations

## 🔧 GPU Optimization Tips

### Check GPU Status
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### Optimize Performance
1. **Use GPU**: Ensure CUDA is properly installed
2. **Batch Processing**: The app automatically optimizes batch sizes
3. **Memory Management**: Close other GPU applications
4. **Video Quality**: Use appropriate resolution (720p recommended)

## 📁 Project Structure

```
Football-Video-Analysis/
├── app.py                      # Main Streamlit application
├── main.py                     # Core analysis logic
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies (for deployment)
├── download_reid_model.py      # ReID model downloader
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── model/
│   └── best.pt                # YOLO detection model
├── tracker/
│   ├── tracker.py             # GPU-optimized tracker
│   └── strongsort_config.yaml # StrongSORT configuration
├── utils/                     # Utility functions
├── ball_assigner/             # Ball assignment logic
├── camera_mouvement_estimator/ # Camera movement analysis
├── speed_and_distance_estimator/ # Player metrics
├── view_transformer/          # View transformation
├── output_videos/             # Processed video output
└── osnet_x0_25_market1501.pt # ReID model
```

## 🐛 Troubleshooting

### Common Issues

#### 1. **CUDA Not Available**
```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. **OpenCV Issues**
```bash
# Reinstall OpenCV
pip uninstall opencv-python
pip install opencv-python
```

#### 3. **Model Files Missing**
```bash
# Download ReID model
python download_reid_model.py

# Ensure YOLO model exists
ls model/best.pt
```

#### 4. **Memory Issues**
- Reduce video resolution
- Close other applications
- Use stub files for faster processing

### Performance Optimization

#### For Better Speed:
1. **Use GPU**: Ensure CUDA is working
2. **Reduce Video Size**: Use 720p instead of 1080p
3. **Use Stub Files**: Enable stub file processing
4. **Close Background Apps**: Free up GPU memory

#### For Better Accuracy:
1. **Higher Resolution**: Use 1080p videos
2. **Full Processing**: Disable stub files
3. **Good Lighting**: Ensure video has good visibility

## 📊 Expected Performance

### Processing Times (with GPU):
- **30-second video (720p)**: ~2-3 minutes
- **1-minute video (720p)**: ~5-7 minutes
- **2-minute video (720p)**: ~10-15 minutes

### Processing Times (CPU only):
- **30-second video (720p)**: ~10-15 minutes
- **1-minute video (720p)**: ~25-35 minutes
- **2-minute video (720p)**: ~50-70 minutes

## 🎯 Best Practices

1. **Video Quality**: Use clear, well-lit football videos
2. **Camera Angle**: Side view or elevated angle works best
3. **File Size**: Keep videos under 200MB for faster processing
4. **GPU Usage**: Monitor GPU memory usage during processing
5. **Backup**: Keep original videos as backup

## 🔄 Updates and Maintenance

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Update Models
```bash
# Re-download ReID model if needed
python download_reid_model.py
```



**⚽ Football Video Analysis Tool | Built with Streamlit and Computer Vision** 