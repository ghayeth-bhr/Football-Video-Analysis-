import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
from main import main
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Football Video Analysis",
    page_icon="⚽",
    layout="wide"
)

# Title and description
st.title("⚽ Football Video Analysis")
st.markdown("Upload a football video to analyze player performance, ball control, and team statistics.")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("📁 Upload Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a football video to analyze"
    )
    
    st.header("⚙️ Settings")
    
    # GPU Status and Fallback System
    import torch
    import psutil
    
    # Detect system capabilities
    gpu_available = torch.cuda.is_available()
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Initialize processing configuration
    processing_config = {
        'device': 'cuda' if gpu_available else 'cpu',
        'batch_size': 16 if gpu_available else 4,
        'half_precision': gpu_available,
        'num_workers': min(cpu_count, 4) if not gpu_available else 0,
        'optimization_level': 'high' if gpu_available else 'balanced'
    }
    
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        gpu_memory_free = gpu_memory_total - gpu_memory_allocated
        cuda_version = torch.version.cuda
        
        st.success(f"🚀 GPU Available: {gpu_name}")
        st.info(f"💾 GPU Memory: {gpu_memory_free:.1f}GB free / {gpu_memory_total:.1f}GB total")
        st.info(f"🔧 CUDA Version: {cuda_version}")
        
        # Optimize batch size based on GPU memory
        if gpu_memory_free < 4:
            processing_config['batch_size'] = 8
            st.warning("⚠️ Limited GPU memory detected. Reduced batch size for stability.")
        elif gpu_memory_free > 8:
            processing_config['batch_size'] = 32
            st.success("✅ High GPU memory available. Optimized batch size for speed.")
        
        st.success("⚡ GPU acceleration enabled for YOLO + StrongSort")
    else:
        st.warning("⚠️ GPU not available. Using CPU fallback system.")
        st.info(f"🖥️ CPU: {cpu_count} cores, {memory_gb:.1f}GB RAM")
        
        # CPU optimization based on system specs
        if cpu_count >= 8 and memory_gb >= 16:
            processing_config['batch_size'] = 8
            processing_config['optimization_level'] = 'high'
            st.success("✅ High-end CPU detected. Optimized for performance.")
        elif cpu_count >= 4 and memory_gb >= 8:
            processing_config['batch_size'] = 4
            processing_config['optimization_level'] = 'balanced'
            st.info("ℹ️ Mid-range CPU detected. Balanced performance settings.")
        else:
            processing_config['batch_size'] = 2
            processing_config['optimization_level'] = 'conservative'
            st.warning("⚠️ Limited CPU resources. Conservative settings for stability.")
    
    # Check if models are ready
    model_path = Path("model/best.pt")
    reid_path = Path("osnet_x0_25_market1501.pt")
    
    if model_path.exists():
        status_icon = "✅" if gpu_available else "🤖"
        st.success(f"{status_icon} YOLO Model: {model_path.name}")
    else:
        st.error(f"❌ YOLO Model not found: {model_path}")
        
    if reid_path.exists():
        status_icon = "✅" if gpu_available else "🔄"
        st.success(f"{status_icon} ReID Model: {reid_path.name}")
    else:
        st.warning(f"⚠️ ReID Model not found: {reid_path}")
        if st.button("📥 Download ReID Model"):
            with st.spinner("Downloading ReID model..."):
                try:
                    import subprocess
                    result = subprocess.run([sys.executable, "download_reid_model.py"], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("✅ ReID model downloaded successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Download error: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # Processing options
    st.subheader("🔧 Processing Options")
    
    # Note: Full processing is always used for best results
    st.info("💡 Full video processing is enabled for optimal analysis results.")
    
    # Advanced settings for CPU optimization
    if not gpu_available:
        st.subheader("🖥️ CPU Optimization")
        
        # Frame skip option for CPU processing
        frame_skip = st.slider("Frame Skip Rate", min_value=1, max_value=5, value=2, 
                              help="Process every Nth frame to speed up CPU processing")
        processing_config['frame_skip'] = frame_skip
        
        # Resolution scaling for CPU
        resolution_scale = st.slider("Resolution Scale", min_value=0.5, max_value=1.0, value=0.75, step=0.25,
                                    help="Scale down video resolution for faster CPU processing")
        processing_config['resolution_scale'] = resolution_scale
        
        # Memory optimization
        memory_optimization = st.checkbox("Memory Optimization", value=True,
                                        help="Reduce memory usage for large videos")
        processing_config['memory_optimization'] = memory_optimization
    
    # Display processing configuration
    st.subheader("⚙️ Current Configuration")
    config_text = f"""
    **Device**: {processing_config['device'].upper()}
    **Batch Size**: {processing_config['batch_size']}
    **Half Precision**: {'Yes' if processing_config['half_precision'] else 'No'}
    **Optimization**: {processing_config['optimization_level'].title()}
    """
    
    if not gpu_available:
        config_text += f"""
    **Frame Skip**: {processing_config.get('frame_skip', 'N/A')}
    **Resolution Scale**: {processing_config.get('resolution_scale', 'N/A')}
    **Memory Optimization**: {'Yes' if processing_config.get('memory_optimization', False) else 'No'}
    """
    
    st.info(config_text)
    


def save_video_as_mp4(frames, output_path, fps=30):
    """Save video frames with robust codec handling to avoid OpenH264 issues"""
    if not frames:
        return False
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    height, width = frames[0].shape[:2]
    
    # Try different codecs in order of preference (avoiding H264/OpenH264 issues)
    # Prioritize web-compatible codecs
    codecs_to_try = [
        ('mp4v', '.mp4'),  # MP4V codec - most reliable, avoids OpenH264
        ('XVID', '.avi'),  # XVID codec - good fallback
        ('MJPG', '.avi'),  # Motion JPEG - another fallback
    ]
    
    for codec, extension in codecs_to_try:
        try:
            # Update file extension if needed
            if not output_path.endswith(extension):
                base_path = os.path.splitext(output_path)[0]
                output_path = base_path + extension
            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"Failed to open VideoWriter with codec {codec}")
                continue
                
            for frame in frames:
                out.write(frame)
            
            out.release()
            print(f"Successfully saved video using {codec} codec: {output_path}")
            return True
            
        except Exception as e:
            print(f"Failed to save with {codec} codec: {e}")
            continue
    
    print("Failed to save video with any codec")
    return False

def create_sample_frames_display(frames, num_samples=8):
    """Create sample frames for browser display without video processing"""
    if not frames:
        return []
    
    # Select evenly distributed sample frames
    total_frames = len(frames)
    if total_frames <= num_samples:
        sample_indices = list(range(total_frames))
    else:
        step = total_frames // num_samples
        sample_indices = [i * step for i in range(num_samples)]
        # Ensure we include the last frame
        if sample_indices[-1] != total_frames - 1:
            sample_indices.append(total_frames - 1)
    
    sample_frames = []
    for idx in sample_indices:
        if idx < len(frames):
            # Convert BGR to RGB for display
            rgb_frame = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            sample_frames.append((rgb_frame, idx))
    
    return sample_frames



# Main content area
if uploaded_file is not None:
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_video_path = tmp_file.name
    
    # Process button
    if st.button("🚀 Process Video", type="primary"):
        # Show processing configuration
        device_status = "🚀 GPU" if processing_config['device'] == 'cuda' else "🖥️ CPU"
        st.info(f"Processing with {device_status} - {processing_config['optimization_level'].title()} optimization")
        
        # Estimate processing time
        if processing_config['device'] == 'cuda':
            estimated_time = "2-10 minutes"
        else:
            if processing_config.get('frame_skip', 1) > 1:
                estimated_time = "5-15 minutes"
            else:
                estimated_time = "10-30 minutes"
        
        st.info(f"⏱️ Estimated processing time: {estimated_time}")
        
        with st.spinner(f"Processing video with {processing_config['device'].upper()}... This may take {estimated_time}."):
            try:
                # Process the video with fallback configuration
                output_video_frames, players_performance_data, teams_data = main(
                    video_path=temp_video_path,
                    model_path=model_path,
                    read_from_stub=False,  # Always use full processing
                    tracks_stub_path=None,
                    camera_movement_stub_path=None,
                    processing_config=processing_config  # Pass the configuration
                )
                
                # Save processed video (fast, no browser optimization)
                output_path = "output_videos/processed_video.mp4"
                video_saved = save_video_as_mp4(output_video_frames, output_path)
                
                if video_saved:
                    st.success("✅ Video processing completed!")
                    
                    # Create tabs for different sections
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Team Statistics", "👥 Player Performance", "🎥 Processed Video", "📈 Detailed Analysis"])
                    
                    with tab1:
                        st.header("Team Statistics")
                        
                        # Team ball control visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart for ball control
                            team_names = list(teams_data.keys())
                            ball_control_values = [teams_data[team]['ball_control'] for team in team_names]
                            
                            fig_pie = px.pie(
                                values=ball_control_values,
                                names=team_names,
                                title="Ball Control Distribution",
                                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            # Bar chart for ball control
                            fig_bar = px.bar(
                                x=team_names,
                                y=ball_control_values,
                                title="Ball Control Percentage",
                                color=team_names,
                                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                            )
                            fig_bar.update_layout(yaxis_title="Ball Control (%)")
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Team statistics table
                        st.subheader("Team Performance Summary")
                        team_df = pd.DataFrame([
                            {
                                'Team': team,
                                'Ball Control (%)': f"{teams_data[team]['ball_control']:.1f}%"
                            }
                            for team in team_names
                        ])
                        st.dataframe(team_df, use_container_width=True)
                    
                    with tab2:
                        st.header("Player Performance")
                        
                        if players_performance_data:
                            # Convert to DataFrame with explicit column selection
                            players_df = pd.DataFrame(list(players_performance_data.values()))
                            
                            # Ensure only the intended columns are present
                            expected_columns = ['player_id', 'max_speed', 'distance covered']
                            if all(col in players_df.columns for col in expected_columns):
                                players_df = players_df[expected_columns]
                            
                            # Filter out rows with None values to show only players with valid data
                            players_df = players_df.dropna()
                            
                            if not players_df.empty:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Max speed distribution
                                    if 'max_speed' in players_df.columns and len(players_df) > 0:
                                        fig_speed = px.histogram(
                                            players_df,
                                            x='max_speed',
                                            title="Player Max Speed Distribution",
                                            nbins=10
                                        )
                                        fig_speed.update_layout(xaxis_title="Max Speed (km/h)", yaxis_title="Number of Players")
                                        st.plotly_chart(fig_speed, use_container_width=True)
                                
                                with col2:
                                    # Distance covered distribution
                                    if 'distance covered' in players_df.columns and len(players_df) > 0:
                                        fig_distance = px.histogram(
                                            players_df,
                                            x='distance covered',
                                            title="Distance Covered Distribution",
                                            nbins=10
                                        )
                                        fig_distance.update_layout(xaxis_title="Distance Covered (m)", yaxis_title="Number of Players")
                                        st.plotly_chart(fig_distance, use_container_width=True)
                                
                                # Player performance table
                                st.subheader("Individual Player Statistics")
                                
                                # Show info about filtered data
                                st.info(f"📊 Showing {len(players_df)} players with complete performance data (filtered out players with missing speed/distance data)")
                                
                                # Format the DataFrame for better display
                                display_df = players_df.copy()
                                
                                # Set player_id as index to remove the default numeric index
                                display_df = display_df.set_index('player_id')
                                
                                # Add performance summary
                                st.markdown("---")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Total Players", len(display_df))
                                
                                with col2:
                                    # Calculate average max speed
                                    if len(display_df) > 0:
                                        avg_speed = display_df['max_speed'].mean()
                                        st.metric("Avg Max Speed", f"{avg_speed:.1f} km/h")
                                    else:
                                        st.metric("Avg Max Speed", "N/A")
                                
                                with col3:
                                    # Calculate average distance
                                    if len(display_df) > 0:
                                        avg_distance = display_df['distance covered'].mean()
                                        st.metric("Avg Distance", f"{avg_distance:.1f} m")
                                    else:
                                        st.metric("Avg Distance", "N/A")
                                
                                st.markdown("---")
                                st.dataframe(display_df, use_container_width=True)
                            else:
                                st.warning("No player performance data available.")
                        else:
                            st.warning("No player performance data available.")
                    
                    with tab3:
                        st.header("Processed Video")
                        
                        # Show sample frames for quick preview
                        st.subheader("🖼️ Sample Frames Preview")
                        if output_video_frames:
                            sample_frames = create_sample_frames_display(output_video_frames, num_samples=8)
                            
                            # Display frames in a grid
                            cols = st.columns(4)  # 4 columns for better layout
                            for i, (rgb_frame, idx) in enumerate(sample_frames):
                                col_idx = i % 4
                                with cols[col_idx]:
                                    st.image(rgb_frame, caption=f"Frame {idx}", use_container_width=True)
                            
                            st.info("💡 These are sample frames from the processed video. Download the full video below for complete playback.")
                        
                        # Download section
                        st.markdown("---")
                        st.subheader("📥 Download Full Video")
                        
                        if os.path.exists(output_path):
                            with open(output_path, "rb") as video_file:
                                video_bytes = video_file.read()
                            
                            st.download_button(
                                label="📥 Download MP4 Video",
                                data=video_bytes,
                                file_name="processed_football_analysis.mp4",
                                mime="video/mp4",
                                help="Download the complete processed video with all annotations"
                            )
                            st.success("✅ Video ready for download! Click the button above to download.")
                        else:
                            st.error("❌ Video file not found. Please try processing again.")
                        
                        # Video info
                        st.markdown("---")
                        st.subheader("📊 Video Information")
                        st.info(f"📹 Video saved as: {output_path}")
                        st.info(f"🎬 Total frames processed: {len(output_video_frames)}")
                        st.info(f"⏱️ Processing completed successfully!")
                    
                    with tab4:
                        st.header("Detailed Analysis")
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                label="Total Players Analyzed",
                                value=len(players_performance_data) if players_performance_data else 0
                            )
                        
                        with col2:
                            avg_ball_control = sum(teams_data[team]['ball_control'] for team in teams_data) / len(teams_data)
                            st.metric(
                                label="Average Ball Control",
                                value=f"{avg_ball_control:.1f}%"
                            )
                        
                        with col3:
                            if players_performance_data:
                                max_speeds = [data['max_speed'] for data in players_performance_data.values() if data['max_speed'] is not None]
                                if max_speeds:
                                    st.metric(
                                        label="Highest Player Speed",
                                        value=f"{max(max_speeds):.1f}"
                                    )
                                else:
                                    st.metric(label="Highest Player Speed", value="N/A")
                            else:
                                st.metric(label="Highest Player Speed", value="N/A")
                        
                        # Additional insights
                        st.subheader("Key Insights")
                        
                        if teams_data:
                            best_team = max(teams_data.keys(), key=lambda x: teams_data[x]['ball_control'])
                            st.info(f"🏆 **{best_team.title()}** had the best ball control with {teams_data[best_team]['ball_control']:.1f}%")
                        
                        if players_performance_data:
                            # Find player with highest speed
                            max_speed_player = None
                            max_speed_value = 0
                            
                            for player_id, data in players_performance_data.items():
                                if data['max_speed'] and data['max_speed'] > max_speed_value:
                                    max_speed_value = data['max_speed']
                                    max_speed_player = player_id
                            
                            if max_speed_player:
                                st.info(f"🏃 **Player {max_speed_player}** achieved the highest speed: {max_speed_value:.1f}")
                else:
                    st.error("❌ Failed to save processed video.")
                
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                st.exception(e)
            finally:
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
else:
    # Welcome message when no file is uploaded
    st.markdown("""
    ### Welcome to Football Video Analysis! 🎯
    
    This application analyzes football videos to provide insights on:
    
    - **Team Performance**: Ball control statistics for each team
    - **Player Analytics**: Individual player speed and distance covered
    - **Visual Analysis**: Processed video with annotations and tracking
    
    ### How to use:
    1. 📁 Upload a football video file (MP4, AVI, MOV, or MKV)
    2. ⚙️ Select your preferred model and settings
    3. 🚀 Click "Process Video" to start analysis
    4. 📊 View results in the interactive tabs
    
    ### Supported Features:
    - Player tracking and identification
    - Ball possession analysis
    - Speed and distance calculations
    - Camera movement compensation
    - Real-time video processing
    """)
    
    # Display sample video if available
    sample_videos = ["input/video0.mp4", "input/video1.mp4", "input/video2.mp4", "input/120.mp4"]
    available_samples = [v for v in sample_videos if os.path.exists(v)]
    
    if available_samples:
        st.subheader("📹 Sample Videos Available")
        st.markdown("You can also test with these sample videos in the `input/` folder:")
        for video in available_samples:
            st.code(video)

# Footer
st.markdown("---")
st.markdown("⚽ Football Video Analysis Tool | Built with Streamlit and Computer Vision") 