import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import cv2
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image, CameraInfo

class RGBDNode(Node):
    def __init__(self, serial_number, rgb_topic_prefix):
        super().__init__(f'{rgb_topic_prefix}_rgbd_node')

        # Create RealSense pipeline for each camera
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)

        # Configure the RealSense pipeline for head or torso camera
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

        # Start streaming
        profile = self.pipeline.start(config)

        # Get stream profiles
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers for aligned depth and RGB images for this camera
        self.rgb_pub = self.create_publisher(Image, f'{rgb_topic_prefix}/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, f'{rgb_topic_prefix}/aligned_depth_to_color/image_raw', 10)
        self.rgb_info_pub = self.create_publisher(CameraInfo, f'{rgb_topic_prefix}/color/camera_info', 10)
        self.depth_info_pub = self.create_publisher(CameraInfo, f'{rgb_topic_prefix}/aligned_depth_to_color/camera_info', 10)

        # Get RealSense camera intrinsics
        self.rgb_intrinsics = color_stream.get_intrinsics()
        self.depth_intrinsics = depth_stream.get_intrinsics()

        # Timer to control the publishing rate (30 Hz)
        self.timer = self.create_timer(1 / 30.0, self.publish_images)

    def publish_images(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        # Get the aligned frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        # Convert RealSense frame to OpenCV format
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert OpenCV images to ROS messages
        color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")

        # Set the header for the color and depth messages
        current_time = self.get_clock().now().to_msg()
        color_msg.header.stamp = current_time
        color_msg.header.frame_id = "camera_color_optical_frame"

        depth_msg.header.stamp = current_time
        depth_msg.header.frame_id = "camera_depth_optical_frame"

        # Create camera info messages
        color_info_msg = self.camera_info_from_intrinsics(self.rgb_intrinsics)
        depth_info_msg = self.camera_info_from_intrinsics(self.depth_intrinsics)

        # Set the header for camera info messages
        color_info_msg.header.stamp = current_time
        color_info_msg.header.frame_id = "camera_color_optical_frame"
        depth_info_msg.header.stamp = current_time
        depth_info_msg.header.frame_id = "camera_depth_optical_frame"

        # Publish the images and camera info
        self.rgb_pub.publish(color_msg)
        self.depth_pub.publish(depth_msg)
        self.rgb_info_pub.publish(color_info_msg)
        self.depth_info_pub.publish(depth_info_msg)

    def camera_info_from_intrinsics(self, intrinsics):
        """Convert RealSense intrinsics to ROS CameraInfo message."""
        camera_info_msg = CameraInfo()
        camera_info_msg.width = intrinsics.width
        camera_info_msg.height = intrinsics.height
        k = [intrinsics.fx, 0, intrinsics.ppx, 0, intrinsics.fy, intrinsics.ppy, 0, 0, 1]
        k = [float(x) for x in k]
        camera_info_msg.k = k
        p = [intrinsics.fx, 0, intrinsics.ppx, 0, 0, intrinsics.fy, intrinsics.ppy, 0, 0, 0, 1, 0]
        p = [float(x) for x in p]
        camera_info_msg.p = p

        # Assuming no distortion for now
        camera_info_msg.distortion_model = 'plumb_bob'
        d = [0, 0, 0, 0, 0]
        d = [float(x) for x in d]
        camera_info_msg.d = d

        return camera_info_msg

def main(args=None):
    rclpy.init(args=args)

    # Serial numbers for the head and torso cameras
    head_serial_number = "239122074149"
    torso_serial_number = "239122070715"

    # Create two nodes for head and torso cameras
    head_node = RGBDNode(head_serial_number, 'head_cam')
    torso_node = RGBDNode(torso_serial_number, 'torso_cam')

    try:
        # Spin both nodes to publish their data
        while rclpy.ok():
            head_node.publish_images()
            torso_node.publish_images()
            rclpy.spin_once(head_node, timeout_sec=0)
            rclpy.spin_once(torso_node, timeout_sec=0)
    except KeyboardInterrupt:
        pass

    # Clean up
    head_node.pipeline.stop()
    torso_node.pipeline.stop()
    head_node.destroy_node()
    torso_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

