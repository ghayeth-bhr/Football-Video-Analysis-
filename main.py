def main(video_path, model_path, read_from_stub=False, tracks_stub_path=None, camera_movement_stub_path=None, processing_config=None):
    from utils import read_video, save_video
    from tracker import Tracker
    from ball_assigner import PlayerBallAssigner
    from camera_mouvement_estimator import CameraMovementEstimator
    from view_transformer import ViewTransformer
    from speed_and_distance_estimator import SpeedAndDistance_Estimator
    import pandas as pd

    import os
    import cv2
    import numpy as np
    # Removed: from boxmot.tracker_zoo import create_tracker (now imported in tracker.py)

    # Apply fallback processing configuration
    if processing_config is None:
        processing_config = {
            'device': 'cuda',
            'batch_size': 16,
            'half_precision': True,
            'optimization_level': 'high'
        }
    
    print(f"Processing with device: {processing_config['device']}")
    print(f"Optimization level: {processing_config['optimization_level']}")
    
    # Read video with potential CPU optimizations
    video_frames = read_video(video_path)
    
    # Apply CPU optimizations if needed
    if processing_config['device'] == 'cpu':
        # Frame skipping for CPU processing
        if processing_config.get('frame_skip', 1) > 1:
            frame_skip = processing_config['frame_skip']
            print(f"Applying frame skip: processing every {frame_skip} frames")
            video_frames = video_frames[::frame_skip]
        
        # Resolution scaling for CPU processing
        if processing_config.get('resolution_scale', 1.0) < 1.0:
            scale = processing_config['resolution_scale']
            print(f"Scaling resolution by factor: {scale}")
            scaled_frames = []
            for frame in video_frames:
                height, width = frame.shape[:2]
                new_height, new_width = int(height * scale), int(width * scale)
                scaled_frame = cv2.resize(frame, (new_width, new_height))
                scaled_frames.append(scaled_frame)
            video_frames = scaled_frames
        
        # Memory optimization for large videos
        if processing_config.get('memory_optimization', False):
            print("Applying memory optimization")
            # Process in chunks to reduce memory usage
            chunk_size = 100  # Process 100 frames at a time
            if len(video_frames) > chunk_size:
                print(f"Large video detected ({len(video_frames)} frames). Processing in chunks.")
    
    tracker = Tracker(model_path, processing_config=processing_config)

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=read_from_stub, stub_path=tracks_stub_path)
    # Get object positions
    tracker.add_position_to_tracks(tracks)
    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=read_from_stub,
        stub_path=camera_movement_stub_path)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # interpolate the ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    # assign ball to a player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    # --- Simplified Team Assignment ---
    # Trust the YOLO model's team classification and use StrongSORT for ID consistency
    # This approach preserves detection quality while maintaining stable IDs
    print("Using YOLO model team classification with StrongSORT ID tracking")
    # --- End Team Assignment ---

    num_frames = len(tracks['ball'])
    for frame_num in range(num_frames):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        teamA_players = tracks['team As'][frame_num]
        teamB_players = tracks['team Bs'][frame_num]

        teamA_player_id = player_assigner.assign_ball_to_player(teamA_players, ball_bbox)
        teamB_player_id = player_assigner.assign_ball_to_player(teamB_players, ball_bbox)

        control_team = {}

        if teamA_player_id != -1:
            teamA_players[teamA_player_id]['has_ball'] = True
            control_team = {'Team': "A"  ,'id': teamA_player_id } 
        elif teamB_player_id != -1:
            teamB_players[teamB_player_id]['has_ball'] = True
            control_team = {'Team': "B"  ,'id': teamB_player_id } 

        else:
            control_team = team_ball_control[-1] if team_ball_control else {'Team': 'None', 'id': -1}

        team_ball_control.append(control_team)
    ## get the final output for teams
    teams_data = {'team A': {}, 'team B': {}}
    team_a_num_frames = sum([entry['Team'] == "A" for entry in team_ball_control])
    team_b_num_frames = sum([entry['Team'] == "B" for entry in team_ball_control])
    total_frames = team_a_num_frames + team_b_num_frames
    # Avoid division by zero
    if total_frames == 0:
        team_a_percent = 0.0
        team_b_percent = 0.0
    else:
        team_a_percent = team_a_num_frames / total_frames
        team_b_percent = team_b_num_frames / total_frames

    team_A_ball_control = team_a_percent * 100
    team_B_ball_control = team_b_percent * 100
    teams_data['team A']['ball_control'] = team_A_ball_control
    teams_data['team B']['ball_control'] = team_B_ball_control

    ## get the final output for players
    players_data = {}
    for team in ['team As', 'team Bs']:
        for frame_dict in tracks[team]:
            for player_id, player_data in frame_dict.items():
                if player_id not in players_data:
                    players_data[player_id] = {'speed': [], 'distance': []}

                # Append only if present
                speed = player_data.get('speed')
                distance = player_data.get('distance')

                if speed is not None:
                    players_data[player_id]['speed'].append(speed)
                if distance is not None:
                    players_data[player_id]['distance'].append(distance)
    players_performance_data = {}
    for player_id, metrics in players_data.items():
        max_speed = max(metrics['speed']) if metrics['speed'] else None
        final_distance = metrics['distance'][-1] if metrics['distance'] else None

        players_performance_data[player_id] = {
            'player_id': player_id,
            'max_speed': max_speed,
            'distance covered': final_distance
        }

    # drawing
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    # Instead of saving video here, return all results for use in web interface
    return output_video_frames, players_performance_data, teams_data

if __name__ == '__main__':
    # For CLI usage, still allow saving video
    output_video_frames, players_performance_data, teams_data = main(
        'input/video3.mp4',
        'model/best.pt',
    )
    from utils import save_video
    save_video(output_video_frames, 'output_videos/video3.avi')