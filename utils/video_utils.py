import cv2
import os

def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    """Save video frames with robust codec handling to avoid OpenH264 issues"""
    if not output_video_frames:
        print("No frames to save")
        return False
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    height, width = output_video_frames[0].shape[:2]
    frame_size = (width, height)
    
    # Try different codecs in order of preference
    codecs_to_try = [
        ('mp4v', '.mp4'),  # MP4V codec - most reliable
        ('XVID', '.avi'),  # XVID codec - good fallback
        ('MJPG', '.avi'),  # Motion JPEG - another fallback
    ]
    
    for codec, extension in codecs_to_try:
        try:
            # Update file extension if needed
            if not output_video_path.endswith(extension):
                base_path = os.path.splitext(output_video_path)[0]
                output_video_path = base_path + extension
            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_video_path, fourcc, 24.0, frame_size)
            
            if not out.isOpened():
                print(f"Failed to open VideoWriter with codec {codec}")
                continue
                
            for frame in output_video_frames:
                out.write(frame)
            
            out.release()
            print(f"Successfully saved video using {codec} codec: {output_video_path}")
            return True
            
        except Exception as e:
            print(f"Failed to save with {codec} codec: {e}")
            continue
    
    print("Failed to save video with any codec")
    return False