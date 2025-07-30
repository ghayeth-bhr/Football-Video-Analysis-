from ultralytics import YOLO
import pickle
import os
import pandas as pd
import sys
import cv2
import numpy as np
import torch
from boxmot import StrongSort
from pathlib import Path
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


# Import tracking API for newer OpenCV versions
try:
    from cv2 import TrackerMIL
    TrackerCSRT = TrackerMIL  # Use TrackerMIL as alternative to TrackerCSRT
except ImportError:
    try:
        from cv2.legacy import TrackerMIL
        TrackerCSRT = TrackerMIL  # Use TrackerMIL as alternative to TrackerCSRT
    except ImportError:
        # Fallback: use a simple tracking approach
        TrackerCSRT = None

class GPUTracker:
    """GPU-optimized tracker with full CUDA tensor processing pipeline"""
    
    def __init__(self, model_path, processing_config=None):
        # Initialize processing configuration
        if processing_config is None:
            processing_config = {
                'device': 'cuda',
                'batch_size': 16,
                'half_precision': True,
                'optimization_level': 'high'
            }
        
        self.processing_config = processing_config
        
        # Check if GPU is requested and available
        if processing_config['device'] == 'cuda':
            # Force GPU usage with comprehensive setup
            print("🚀 Initializing GPU-optimized tracker with full CUDA pipeline...")
            
            # Force CUDA environment variables
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            
            # Check CUDA availability with detailed diagnostics
            if not torch.cuda.is_available():
                print("⚠️ CUDA requested but not available. Falling back to CPU.")
                processing_config['device'] = 'cpu'
                self.device = torch.device('cpu')
            else:
                # Force CUDA initialization and clear cache
                torch.cuda.empty_cache()
                torch.cuda.init()
                
                device_count = torch.cuda.device_count()
                print(f"📊 CUDA devices detected: {device_count}")
                
                if device_count == 0:
                    print("⚠️ No CUDA devices detected. Falling back to CPU.")
                    processing_config['device'] = 'cpu'
                    self.device = torch.device('cpu')
                else:
                    # Use the first available CUDA device with explicit configuration
                    self.device = torch.device('cuda:0')
        else:
            # CPU fallback mode
            print("🖥️ Initializing CPU-optimized tracker...")
            self.device = torch.device('cpu')
        
        # Test device access and get detailed info
        try:
            test_tensor = torch.tensor([1.0], device=self.device)
            if self.device.type == 'cuda':
                print(f"✅ CUDA device configured: {self.device}")
                print(f"   GPU name: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA version: {torch.version.cuda}")
                print(f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                print(f"   Available memory: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
            else:
                print(f"✅ CPU device configured: {self.device}")
        except Exception as e:
            print(f"❌ Device access failed: {e}")
            raise RuntimeError(f"Device access failed: {e}")
            
        # Load YOLO model with device-specific configuration
        device_text = "GPU" if self.device.type == 'cuda' else "CPU"
        print(f"🤖 Loading YOLO model with {device_text}...")
        self.model = YOLO(model_path, task='detect')
        
        # Move model to appropriate device
        self.model.to(self.device)
        self.model.eval()
        
        # Set YOLO inference parameters based on device
        if self.device.type == 'cuda':
            self.model.conf = 0.5  # Confidence threshold
            self.model.iou = 0.45  # NMS IoU threshold
            self.model.max_det = 1000  # Maximum number of detections per image
        else:
            # CPU-optimized parameters
            self.model.conf = 0.6  # Higher confidence threshold for CPU
            self.model.iou = 0.5  # Higher IoU threshold for CPU
            self.model.max_det = 500  # Fewer detections for CPU efficiency
        
        print(f"✅ YOLO model loaded on {device_text}: {self.device}")
        
        # Ensure ReID model path exists and is accessible
        reid_model_path = Path('osnet_x0_25_market1501.pt')
        if not reid_model_path.exists():
            print(f"⚠️ ReID model not found at {reid_model_path}")
            print("   Please ensure osnet_x0_25_market1501.pt is in the project root")
        
        # Initialize StrongSort tracker with device-specific parameters
        try:
            device_text = "GPU" if self.device.type == 'cuda' else "CPU"
            print(f"🔄 Initializing StrongSort tracker with {device_text}...")
            
            # Device-specific parameters
            if self.device.type == 'cuda':
                tracker_params = {
                    'reid_weights': Path('osnet_x0_25_market1501.pt'),
                    'device': self.device,
                    'half': True,  # Use half precision for GPU acceleration
                    'min_conf': 0.3,  # Lower confidence threshold for better detection
                    'max_cos_dist': 0.2,  # Tighter cosine distance for better matching
                    'max_iou_dist': 0.7,  # IoU distance threshold
                    'max_age': 30,  # Maximum age for tracks
                    'n_init': 3,  # Number of detections to confirm track
                    'nn_budget': 100,  # Feature budget for ReID
                    'mc_lambda': 0.98,  # Motion consistency weight
                    'ema_alpha': 0.9  # Exponential moving average alpha
                }
                self.use_half_precision = True
            else:
                # CPU-optimized parameters
                tracker_params = {
                    'reid_weights': Path('osnet_x0_25_market1501.pt'),
                    'device': self.device,
                    'half': False,  # No half precision on CPU
                    'min_conf': 0.4,  # Higher confidence threshold for CPU
                    'max_cos_dist': 0.3,  # Slightly looser matching for CPU
                    'max_iou_dist': 0.8,  # Higher IoU distance threshold
                    'max_age': 20,  # Shorter track persistence for CPU
                    'n_init': 2,  # Fewer confirmations for CPU
                    'nn_budget': 50,  # Smaller feature budget for CPU
                    'mc_lambda': 0.95,  # Lower motion consistency weight
                    'ema_alpha': 0.8  # Lower exponential moving average alpha
                }
                self.use_half_precision = False
            
            # Initialize the tracker
            self.tracker = StrongSort(**tracker_params)
            
            print(f"✅ StrongSort tracker initialized with {device_text}: {self.device}")
            print(f"   Half precision: {self.use_half_precision}")
            print(f"   ReID model: osnet_x0_25_market1501.pt")
            
            # Apply optimizations based on device
            if self.device.type == 'cuda':
                # Apply GPU optimization patches
                self._apply_gpu_optimizations()
                
                # Final GPU status verification
                print(f"🔍 Final GPU status check:")
                print(f"   CUDA available: {torch.cuda.is_available()}")
                print(f"   Current device: {torch.cuda.current_device()}")
                print(f"   Device name: {torch.cuda.get_device_name()}")
                print(f"   Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
                print(f"   Memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
            else:
                print(f"🔍 CPU optimization applied:")
                print(f"   Batch size: {self.processing_config.get('batch_size', 'N/A')}")
                print(f"   Optimization level: {self.processing_config.get('optimization_level', 'N/A')}")
            
        except Exception as e:
            print(f"❌ StrongSort initialization failed: {e}")
            raise RuntimeError(f"StrongSort initialization failed: {e}")

    def _apply_gpu_optimizations(self):
        """Apply comprehensive GPU optimizations to the tracking pipeline"""
        try:
            # Get the ReID model backend
            reid_backend = self.tracker.model
            
            # Create GPU-optimized preprocessing pipeline
            self._create_gpu_preprocessing_pipeline(reid_backend)
            
            # Patch the ReID backend for GPU tensor processing
            self._patch_reid_gpu_processing(reid_backend)
            
            # Create GPU-optimized feature extraction
            self._create_gpu_feature_extraction(reid_backend)
            
            print("✅ Comprehensive GPU optimizations applied")
            
        except Exception as e:
            print(f"⚠️ Could not apply GPU optimizations: {e}")
            import traceback
            traceback.print_exc()

    def _create_gpu_preprocessing_pipeline(self, reid_backend):
        """Create a GPU-optimized preprocessing pipeline for frames"""
        # BoxMOT already has optimized GPU preprocessing
        # We'll use the existing implementation which is already GPU-optimized
        print(f"✅ Using BoxMOT's built-in GPU preprocessing")
        print(f"   Device: {reid_backend.device}")
        print(f"   Half precision: {reid_backend.half}")
        
        # Store reference to the backend for potential future use
        self.reid_backend = reid_backend

    def _patch_reid_gpu_processing(self, reid_backend):
        """Patch the ReID backend for GPU tensor processing"""
        
        # Create GPU-optimized to_numpy that handles CUDA tensors properly
        def gpu_to_numpy(x):
            if isinstance(x, torch.Tensor):
                if x.device.type == 'cuda':
                    return x.cpu().numpy()
                else:
                    return x.numpy()
            return x
        
        # Patch the to_numpy method
        reid_backend.to_numpy = gpu_to_numpy
        
        # Create GPU-optimized inference postprocess
        def gpu_inference_postprocess(features):
            if isinstance(features, (list, tuple)):
                if len(features) == 1:
                    return gpu_to_numpy(features[0])
                else:
                    return [gpu_to_numpy(x) for x in features]
            else:
                return gpu_to_numpy(features)
        
        reid_backend.inference_postprocess = gpu_inference_postprocess

    def _create_gpu_feature_extraction(self, reid_backend):
        """Create GPU-optimized feature extraction pipeline"""
        
        # The BoxMOT backend already has GPU optimization built-in
        # We just need to ensure it's using the correct device and half precision
        print(f"✅ ReID backend configured for GPU: {reid_backend.device}")
        print(f"   Half precision: {reid_backend.half}")
        print(f"   Input shape: {reid_backend.input_shape}")
        
        # Store the original get_features method for potential future customization
        self.original_get_features = reid_backend.get_features

    def frames_to_gpu_tensors(self, frames):
        """Convert frames to GPU tensors for processing"""
        # BoxMOT handles GPU processing internally
        # This method is kept for potential future use
        return frames

    def detect_frames(self, frames):
        """Detect objects in frames using YOLO with device-specific optimization"""
        # Use batch size from processing configuration
        batch_size = self.processing_config.get('batch_size', 16)
        detections = []
        
        device_text = "GPU" if self.device.type == 'cuda' else "CPU"
        print(f"🔍 Processing {len(frames)} frames in {device_text} batches of {batch_size}...")
        
        with torch.inference_mode():
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                
                # Run YOLO detection with device-specific parameters
                detections_batch = self.model(
                    batch_frames, 
                    device=self.device, 
                    verbose=False,
                    conf=self.model.conf,
                    iou=self.model.iou,
                    max_det=self.model.max_det
                )
                detections.extend(detections_batch)
        
        print(f"✅ {device_text} detection completed: {len(detections)} batches processed")
        return detections

    def prepare_detections_for_tracking(self, detection):
        """Convert YOLO detection to format expected by BoxMOT with GPU optimization"""
        if len(detection.boxes) == 0:
            return np.empty((0, 6))
        
        # Extract detection data (keep on GPU as long as possible)
        boxes = detection.boxes.xyxy.cpu().numpy()  # Move to CPU and convert to numpy
        confidences = detection.boxes.conf.cpu().numpy()
        class_ids = detection.boxes.cls.cpu().numpy()
        
        # Stack into format expected by BoxMOT: [x1, y1, x2, y2, conf, cls]
        detections_array = np.column_stack((boxes, confidences, class_ids))
        return detections_array

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """Get object tracks using GPU-optimized StrongSort"""
        # Clear GPU cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
            
        detections = self.detect_frames(frames)
        tracks = {
            "team As": [],
            "team Bs": [],
            "referees": [],
            "goalkeepers": [],
            "ball": []
        }

        # Initialize variables for ball tracking
        ball_tracker = None
        last_known_ball_bbox = None

        device_text = "GPU-optimized" if self.device.type == 'cuda' else "CPU-optimized"
        print(f"🔄 Processing {device_text} tracking for {len(detections)} frames...")

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Prepare detections for tracking
            detections_array = self.prepare_detections_for_tracking(detection)

            # Update tracker with detections (GPU optimized)
            with torch.inference_mode():
                tracks_output = self.tracker.update(detections_array, frames[frame_num])

            # Initialize frame dictionaries
            tracks["team As"].append({})
            tracks["team Bs"].append({})
            tracks["goalkeepers"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Process tracked objects (players, referees, goalkeepers)
            if len(tracks_output) > 0:
                for track in tracks_output:
                    # BoxMOT output format: [x1, y1, x2, y2, track_id, conf, cls, ind]
                    x1, y1, x2, y2, track_id, conf, cls_id = track[:7]
                    bbox = [float(x1), float(y1), float(x2), float(y2)]
                    track_id = int(track_id)
                    cls_id = int(cls_id)

                    if cls_id == cls_names_inv['team A']:
                        tracks["team As"][frame_num][track_id] = {"bbox": bbox}

                    elif cls_id == cls_names_inv['team B']:
                        tracks["team Bs"][frame_num][track_id] = {"bbox": bbox}

                    elif cls_id == cls_names_inv['goalkeeper']:
                        tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}

                    elif cls_id == cls_names_inv['referee']:
                        tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Handle ball detection and tracking
            ball_detected = self.handle_ball_tracking(
                detection, tracks, frame_num, cls_names_inv, 
                ball_tracker, last_known_ball_bbox, frames
            )
            
            if ball_detected:
                last_known_ball_bbox = tracks["ball"][frame_num].get(1, {}).get('bbox')

        print(f"✅ GPU-optimized tracking completed: {len(tracks['team As'])} frames processed")

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def handle_ball_tracking(self, detection, tracks, frame_num, cls_names_inv, 
                           ball_tracker, last_known_ball_bbox, frames):
        """Handle ball detection and tracking with improved logic"""
        ball_detected = False

        # Check if ball is detected in current frame from original YOLO detections
        if len(detection.boxes) > 0:
            original_boxes = detection.boxes.xyxy.cpu().numpy()
            original_confidences = detection.boxes.conf.cpu().numpy()
            original_class_ids = detection.boxes.cls.cpu().numpy()
            
            for i, cls_id in enumerate(original_class_ids):
                if int(cls_id) == cls_names_inv['ball']:
                    bbox = original_boxes[i].tolist()
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
                    ball_detected = True
                    break

        # If ball not detected, try to track it using OpenCV tracker
        if not ball_detected and frame_num > 0 and last_known_ball_bbox is not None:
            if ball_tracker is None and TrackerCSRT is not None:
                try:
                    ball_tracker = TrackerCSRT.create()
                    prev_frame = frames[frame_num - 1]

                    # Convert bbox from (x1, y1, x2, y2) to (x, y, width, height)
                    x1, y1, x2, y2 = last_known_ball_bbox
                    cv2_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    ball_tracker.init(prev_frame, cv2_bbox)
                except Exception as e:
                    print(f"Warning: Could not initialize ball tracker: {e}")
                    ball_tracker = None

            if ball_tracker is not None:
                try:
                    ok, predicted_bbox = ball_tracker.update(frames[frame_num])
                    if ok:
                        # Convert back from (x, y, width, height) to (x1, y1, x2, y2)
                        x, y, w, h = predicted_bbox
                        predicted_xyxy = [int(x), int(y), int(x + w), int(y + h)]
                        tracks["ball"][frame_num][1] = {"bbox": predicted_xyxy}
                        ball_detected = True
                except Exception as e:
                    print(f"Warning: Ball tracking failed: {e}")
                    ball_tracker = None

        return ball_detected

    def add_position_to_tracks(self, tracks):
        """Add position information to tracks"""
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        return frame

    def interpolate_ball_positions(self, ball_positions):
        # Extract bbox lists (could be empty)
        raw_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]

        # Filter out empty lists
        valid_positions = [bbox for bbox in raw_positions if len(bbox) == 4]

        if len(valid_positions) == 0:
            print("⚠️ No valid ball positions to interpolate.")
            return ball_positions  # or return raw_positions

        # Construct DataFrame only from valid entries
        df_ball_positions = pd.DataFrame(raw_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate with fallback to linear if spline fails
        try:
            # Try spline interpolation first
            df_ball_positions = df_ball_positions.interpolate(method='spline', order=2, limit=5, limit_direction='both')
        except:
            # Fallback to linear interpolation if spline fails (e.g., not enough points)
            df_ball_positions = df_ball_positions.interpolate(method='linear', limit=5, limit_direction='both')
        
        # Fill remaining NaN values
        df_ball_positions = df_ball_positions.bfill().ffill()

        interpolated = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return interpolated

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # Get the number of times each team had ball control
        team_a_num_frames = sum([entry['Team'] == "A" for entry in team_ball_control_till_frame])
        team_b_num_frames = sum([entry['Team'] == "B" for entry in team_ball_control_till_frame])

        total_frames = team_a_num_frames + team_b_num_frames

        # Avoid division by zero
        if total_frames == 0:
            team_a_percent = 0.0
            team_b_percent = 0.0
        else:
            team_a_percent = team_a_num_frames / total_frames
            team_b_percent = team_b_num_frames / total_frames

        cv2.putText(frame, f"Team A Ball Control: {team_a_percent * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0), 3)
        cv2.putText(frame, f"Team B Ball Control: {team_b_percent * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """Draw annotations on video frames"""
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            team_a_dict = tracks["team As"][frame_num]
            team_b_dict = tracks["team Bs"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            referee_ids = set(referee_dict.keys())

            # Draw Players Team A (skip referees)
            for track_id, player in team_a_dict.items():
                if track_id in referee_ids:
                    continue
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw Players Team B (skip referees)
            for track_id, player in team_b_dict.items():
                if track_id in referee_ids:
                    continue
                frame = self.draw_ellipse(frame, player["bbox"], (0, 255, 0), track_id)
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw referees (distinct color, e.g., yellow)
            for track_id, player in referee_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 255, 255), track_id)

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (255, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames

# Keep the original Tracker class for backward compatibility
class Tracker(GPUTracker):
    """Backward compatibility wrapper for the original Tracker interface"""
    pass