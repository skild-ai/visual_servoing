#!/usr/bin/env python3
# capture images with the given camera serial numbers, exposure white balancing etc calibrated for the camera properly

import os
import cv2
import numpy as np
import pyrealsense2 as rs
import time

username = os.environ.get("USER")
# Camera serial numbers
head_cam_serial = "239122074149"
torso_cam_serial = "239122070715"

# Base output directory
output_dir_base = "/home/{}/poorvi/data/detection_bin".format(username)
os.makedirs(output_dir_base, exist_ok=True)

# Function to get the next trajectory directory
def get_next_traj_dir(base_dir):
    # Get all existing directories that start with "traj"
    traj_dirs = [d for d in os.listdir(base_dir) if d.startswith("traj")]
    # Find the next available trajectory number
    if traj_dirs:
        traj_indices = [int(d.replace("traj", "")) for d in traj_dirs if d.replace("traj", "").isdigit()]
        next_traj_index = max(traj_indices) + 1 if traj_indices else 0
    else:
        next_traj_index = 0
    # Return the new trajectory directory path
    return os.path.join(base_dir, f"traj{next_traj_index}")

# Helper function to save camera intrinsics
def save_camera_intrinsics(pipeline, serial_no, output_dir):
    # Get the pipeline profile and intrinsics
    pipeline_profile = pipeline.get_active_profile()
    color_stream_profile = pipeline_profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream_profile.as_video_stream_profile().get_intrinsics()

    # Save intrinsics to a file
    intrinsics_file = os.path.join(output_dir, "intrinsics.txt")
    with open(intrinsics_file, "w") as f:
        f.write(f"Camera Serial Number: {serial_no}\n")
        f.write(f"Width: {color_intrinsics.width}\n")
        f.write(f"Height: {color_intrinsics.height}\n")
        f.write(f"PPX: {color_intrinsics.ppx}\n")
        f.write(f"PPY: {color_intrinsics.ppy}\n")
        f.write(f"Fx: {color_intrinsics.fx}\n")
        f.write(f"Fy: {color_intrinsics.fy}\n")
        f.write(f"Distortion Model: {color_intrinsics.model}\n")
        f.write(f"Coeffs: {color_intrinsics.coeffs}\n")

def capture_images(serial_no, rgb_dir, depth_dir, index):
    # Initialize the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_no)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # Start the pipeline
    profile = pipeline.start(config)
    device = profile.get_device()
    color_sensor = device.query_sensors()[1]  # Index 1 corresponds to the color sensor
    color_sensor.set_option(rs.option.enable_auto_white_balance, 0) # disable auto white balance
    color_sensor.set_option(rs.option.enable_auto_exposure, 0) 
    color_sensor.set_option(rs.option.white_balance, 3500)
    color_sensor.set_option(rs.option.exposure, 200) 
   # color_sensor.set_option(rs.option.brightness, 25)
    aligned_stream = rs.align(rs.stream.color)

    try:
        frames = pipeline.wait_for_frames()
        aligned_frames = aligned_stream.process(frames)
        
        # Get the RGB and depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print(f"Could not capture frames from camera {serial_no}")
            return
        
        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Save RGB and depth images
        rgb_image_path = os.path.join(rgb_dir, f"{index}.png")
        depth_image_path = os.path.join(depth_dir, f"{index}.png")
        cv2.imwrite(rgb_image_path, color_image)  # Save the RGB image
        cv2.imwrite(depth_image_path, depth_image)
        print(f"Images {index}.png saved for camera {serial_no}")

    finally:
        # Stop the pipeline
        pipeline.stop()

# Function to continuously capture images every 1/30 second
def continuous_capture():
    index = 0

    # Get the next available trajectory directory
    next_traj_dir = get_next_traj_dir(output_dir_base)

    # Create directories for head and torso cameras in the trajectory directory
    head_cam_rgb_dir = os.path.join(next_traj_dir, "head_cam", "rgb")
    head_cam_depth_dir = os.path.join(next_traj_dir, "head_cam", "depth")
    torso_cam_rgb_dir = os.path.join(next_traj_dir, "torso_cam", "rgb")
    torso_cam_depth_dir = os.path.join(next_traj_dir, "torso_cam", "depth")

    # Create directories if they do not exist
    os.makedirs(head_cam_rgb_dir, exist_ok=True)
    os.makedirs(head_cam_depth_dir, exist_ok=True)
    os.makedirs(torso_cam_rgb_dir, exist_ok=True)
    os.makedirs(torso_cam_depth_dir, exist_ok=True)

    # Save camera intrinsics for both cameras
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(head_cam_serial)
    pipeline_profile = pipeline.start(config)
    save_camera_intrinsics(pipeline, head_cam_serial, head_cam_rgb_dir)
    pipeline.stop()

    config.enable_device(torso_cam_serial)
    pipeline_profile = pipeline.start(config)
    save_camera_intrinsics(pipeline, torso_cam_serial, torso_cam_rgb_dir)
    pipeline.stop()

    while True:
        # Capture images for both head and torso cameras
        capture_images(head_cam_serial, head_cam_rgb_dir, head_cam_depth_dir, index)
        capture_images(torso_cam_serial, torso_cam_rgb_dir, torso_cam_depth_dir, index)

        # Wait for 1/30 second (for 30 FPS)
        time.sleep(1 / 30.0)

        index += 1

# Start continuous capture
continuous_capture()

