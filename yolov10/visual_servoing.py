#!/usr/bin/env python3
# does visual servoing 9(with start and end action service calls)
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, Point, PoseStamped, Quaternion, Pose
from std_msgs.msg import String
from std_srvs.srv import SetBool
from cv_bridge import CvBridge
from ultralytics import YOLOv10
import os
import math
import torch
import tf_transformations
from tf2_ros import Buffer, TransformListener, TransformException
from std_msgs.msg import Float32

# Define label mapping for the classes
# label_mapping = {
#     "spoon": 0,
#     "cup": 1,
#     "cabinet": 2,
#     "sink": 3,
#     "plate": 4
# }

def get_next_traj_dir(base_dir):
    traj_dirs = [d for d in os.listdir(base_dir) if d.startswith("traj")]
    next_traj_index = max([int(d.replace("traj", "")) for d in traj_dirs if d.replace("traj", "").isdigit()], default=-1) + 1
    return os.path.join(base_dir, f"traj{next_traj_index}")


class DualCameraProcessorAndFollower(Node):
    def __init__(self, output_dir):
        super().__init__('dual_camera_processor_and_follower')
        print("Initializing DualCameraProcessorAndFollower")

        # Set up directory for saving images
        self.output_dir = output_dir
        self.rgb_dir = os.path.join(self.output_dir, "rgb")
        self.depth_dir = os.path.join(self.output_dir, "depth")
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.rgb_dir, exist_ok=True)

        # Visual servoing flags
        self.visual_servo_started = False
        self.end_servoing_called = False

        # Initialize object detection state
        self.object_detected = False
        self.object_position = None

        # TF buffer and listener for transform
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Parameters for movement control
        # self.linear_speed_factor_x = 0.5
        # self.linear_speed_factor_y = 1.2 
        # self.angular_speed_factor = 0.2
        self.linear_speed_factor_x = 0.4
        self.linear_speed_factor_y = 1 
        self.angular_speed_factor = 0.1

        # Goal tolerances for x, y, and angular z
        self.goal_tolerance_x = None
        self.goal_tolerance_y_left = None
        self.goal_tolerance_y_right = None
        self.goal_tolerance_y = None
        self.goal_tolerance_ang_z = 0.2  # Angular tolerance in radians -> 6 degrees approx
        self.desired_orientation_yaw = None

        self.use_odometry = False

        # # Get table and sink orientation in the base_link frame
        # if self.use_odometry:
        #     self.get_transformed_positions()

        # YOLO model setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        # self.model = YOLOv10("../test/multiclass_m.pt").to(device)
        # self.spoon_model = YOLOv10("../models/weights_m/spoon.pt").to(device)
        # self.spoon_model = YOLOv10("../models/weights_m/spatula.pt").to(device)
        self.spoon_model = YOLOv10("../models/weights_m/spoon_combined.pt").to(device)
        self.cabinet_model = YOLOv10("../models/weights_m/cabinet.pt").to(device)
        self.cup_model = YOLOv10("../models/weights_m/cup.pt").to(device)
        self.plate_model = YOLOv10("../models/weights_m/plate.pt").to(device)
        self.sink_model = YOLOv10("../models/weights_m/sink.pt").to(device)
        #labelled
        print("models loaded")

        # Create a service to start and end visual servoing
        self.start_visual_servo_service = self.create_service(SetBool, 'start_visual_servo', self.start_visual_servo_callback)
        self.end_visual_servo_client = self.create_client(SetBool, 'end_visual_servo')

        # Subscribe to category topic to receive category label
        self.category_sub = self.create_subscription(String, '/visual_servo_category', self.category_callback, 10)

        # Topics and camera setup
        self.head_rgb_topic = '/head_cam/color/image_raw'
        self.torso_rgb_topic = '/torso_cam/color/image_raw'
        self.head_depth_topic = '/head_cam/aligned_depth_to_color/image_raw'
        self.torso_depth_topic = '/torso_cam/aligned_depth_to_color/image_raw'
        self.head_rgb_info_topic = '/head_cam/color/camera_info'
        self.torso_rgb_info_topic = '/torso_cam/color/camera_info'

        # ROS subscribers for head and torso cameras
        self.head_rgb_sub = self.create_subscription(Image, self.head_rgb_topic, self.head_rgb_callback, 10)
        self.torso_rgb_sub = self.create_subscription(Image, self.torso_rgb_topic, self.torso_rgb_callback, 10)
        self.head_depth_sub = self.create_subscription(Image, self.head_depth_topic, self.head_depth_callback, 10)
        self.torso_depth_sub = self.create_subscription(Image, self.torso_depth_topic, self.torso_depth_callback, 10)
        self.head_rgb_info_sub = self.create_subscription(CameraInfo, self.head_rgb_info_topic, self.head_rgb_info_callback, 10)
        self.torso_rgb_info_sub = self.create_subscription(CameraInfo, self.torso_rgb_info_topic, self.torso_rgb_info_callback, 10)

        # Publisher for /cmd_vel to send velocity commands and publishers for detection data
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.detection_3d_pub = self.create_publisher(Point, '/detection_3d', 10)
        self.detect_publisher = self.create_publisher(String, '/detect', 10)

        # Bridge for OpenCV
        self.bridge = CvBridge()

        # Camera parameters
        self.head_rgb_intrinsics = None
        self.torso_rgb_intrinsics = None
        self.head_depth_intrinsics = None
        self.torso_depth_intrinsics = None
        self.head_extrinsics = {
            'R': np.array([[0, -0.707, 0.707], [-1, 0, 0], [0, -0.707, -0.707]]),
            't': np.array([0.04, 0.03, 0.71])
        }
        self.torso_extrinsics = {
            'R': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
            't': np.array([0.07, 0.03, 0.56])
        }

        # Image and depth data
        self.head_rgb_image = None
        self.torso_rgb_image = None
        self.head_depth_image = None
        self.torso_depth_image = None
        self.index = 0
        self.label_name = None

        self.table_orientation_yaw = None
        self.sink_orientation_yaw = None

        self.poses = {
            "table": {
                "position": np.array([13.687, 9.590, 0.0]),
                "orientation": [0.0, 0.0, 0.605, 0.798]
            },
            "sink": {
                "position": np.array([13.632, 9.4703, 0.0]),
                "orientation": [0.0, 0.0, -0.790, 0.613]
            }
        }
        self.desired_orientation_pub = self.create_publisher(Float32, '/desired_orientation_yaw', 10)
        self.ang_z_thres = 0.05


    def start_visual_servo_callback(self, request, response):
        """Callback to start visual servoing upon receiving a service call."""
        if request.data:
            self.visual_servo_started = True
            self.end_servoing_called = False
            response.success = True
            response.message = 'Visual servoing started'
            self.get_logger().info(f'Started visual servoing for category: {self.label_name}')
        else:
            self.visual_servo_started = False
            response.success = False
            response.message = 'Visual servoing not started'
            self.get_logger().info('Visual servoing stopped')

        return response

    def category_callback(self, msg):
        """Callback to set the label name from the category topic."""
        self.label_name = msg.data.lower()
        self.get_logger().info(f"Received category: {self.label_name}")

    def head_rgb_callback(self, msg):
        self.head_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.run_inference()

    def torso_rgb_callback(self, msg):
        self.torso_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.run_inference()

    def head_depth_callback(self, msg):
        self.head_depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

    def torso_depth_callback(self, msg):
        self.torso_depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

    def head_rgb_info_callback(self, msg):
        if not self.head_rgb_intrinsics:
            self.head_rgb_intrinsics = self.extract_intrinsics(msg)

    def torso_rgb_info_callback(self, msg):
        if not self.torso_rgb_intrinsics:
            self.torso_rgb_intrinsics = self.extract_intrinsics(msg)

    def extract_intrinsics(self, camera_info_msg):
        return {
            'fx': camera_info_msg.k[0],
            'fy': camera_info_msg.k[4],
            'cx': camera_info_msg.k[2],
            'cy': camera_info_msg.k[5]
        }

    def transform_pose(self, transform, pose_label):
        # Get translation and inverse rotation from global_frame_2 to base_link
        trans = transform.transform.translation
        base_link_translation = np.array([trans.x, trans.y, trans.z])

        rot = transform.transform.rotation
        base_link_quaternion = [rot.x, rot.y, rot.z, rot.w]
        inverse_rotation_matrix = tf_transformations.quaternion_matrix(base_link_quaternion).T[:3, :3]
        object_position = self.poses[pose_label]["position"]
        object_orientation = self.poses[pose_label]["orientation"]

        relative_position = object_position - base_link_translation
        position_in_base_link = inverse_rotation_matrix.dot(relative_position)
        object_roll, object_pitch, object_yaw = tf_transformations.euler_from_quaternion(object_orientation)
        base_link_roll, base_link_pitch, base_link_yaw = tf_transformations.euler_from_quaternion(base_link_quaternion)
        roll_in_base_link = object_roll - base_link_roll
        pitch_in_base_link = object_pitch - base_link_pitch
        yaw_in_base_link = object_yaw - base_link_yaw


        return {
            'position': (round(position_in_base_link[0], 2), 
                        round(position_in_base_link[1], 2), 
                        round(position_in_base_link[2], 2)),
            'orientation': (round(roll_in_base_link, 2), round(pitch_in_base_link, 2), round(yaw_in_base_link, 2))
        }

    def get_transformed_positions(self):
        """Fetches the latest orientations for table and sink in the base_link frame."""
        timeout_duration = rclpy.duration.Duration(seconds=0.5)
        try:
            # Lookup transform from global_frame_2 to base_link
            # transform = self.tf_buffer.lookup_transform("global_frame_2", "base_link", rclpy.time.Time(), timeout_duration)
            transform = self.tf_buffer.lookup_transform("odom", "base_link", rclpy.time.Time(), timeout_duration)

            # Transform and update the orientations for table and sink
            table_transformed = self.transform_pose(transform, "table")
            sink_transformed = self.transform_pose(transform, "sink")
            
            # Store yaw for table and sink in base_link frame
            self.table_orientation_yaw = table_transformed['orientation'][2]
            self.sink_orientation_yaw = sink_transformed['orientation'][2]

        except TransformException as ex:
            self.get_logger().info(f"Could not transform global_frame_2 to base_link: {ex}")


    def run_inference(self):
        """Inference function to detect object and control robot movement."""
        # Only proceed if visual servoing is active and label_name is set
        if not self.visual_servo_started or not self.label_name:
            return

        if self.head_rgb_image is None or self.head_depth_image is None or self.torso_rgb_image is None or self.torso_depth_image is None:
            return

        closest_detection = None
        min_distance = float('inf')

        if self.use_odometry:
            self.get_transformed_positions()

        for camera, rgb_image, depth_image, intrinsics, extrinsics in [
            ('head', self.head_rgb_image, self.head_depth_image, self.head_rgb_intrinsics, self.head_extrinsics),
            ('torso', self.torso_rgb_image, self.torso_depth_image, self.torso_rgb_intrinsics, self.torso_extrinsics)
        ]:
            if self.label_name == 'spoon':
                self.model = self.spoon_model
                self.goal_tolerance_x = 0.5  # Distance threshold for x-axis (forward movement)
                self.goal_tolerance_y_left = 0.1  
                self.goal_tolerance_y_right = 0.15
                self.goal_tolerance_y = 0.1
                self.desired_orientation_yaw = self.table_orientation_yaw

                
            if self.label_name == 'sink':
                self.model = self.sink_model
                self.goal_tolerance_x = 1.1  # Distance threshold for x-axis (forward movement)
                self.goal_tolerance_y_left = 0.1  
                self.goal_tolerance_y_right = 0.3
                self.goal_tolerance_y = 0.1
                self.desired_orientation_yaw = self.sink_orientation_yaw

            # if self.label_name == 'cabinet':
            #     self.model = self.cabinet_model

            if self.label_name == 'cup':
                self.model = self.cup_model
                self.goal_tolerance_x = 0.5  # Distance threshold for x-axis (forward movement)
                self.goal_tolerance_y_left = 0.1  
                self.goal_tolerance_y_right = 0.3
                self.goal_tolerance_y = 0.1
                self.desired_orientation_yaw = self.table_orientation_yaw

            if self.label_name == 'plate':
                self.model = self.plate_model
                self.goal_tolerance_x = 0.7  # Distance threshold for x-axis (forward movement)
                self.goal_tolerance_y_left = 0.1  
                self.goal_tolerance_y_right = 0.3
                self.goal_tolerance_y = 0.1
                self.desired_orientation_yaw = self.table_orientation_yaw

            results = self.model.predict(source=rgb_image, conf=0.35)

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    class_name = self.model.names[class_id].lower()

                    if class_name == self.label_name:
                        x1, y1, x2, y2 = box.xyxy[0]
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        depth = depth_image[center_y, center_x] * 0.001

                        if 0.001 < depth < min_distance:
                            min_distance = depth
                            closest_detection = (box, extrinsics, intrinsics, depth, camera)

        if closest_detection:
            box, extrinsics, intrinsics, depth, camera = closest_detection

            if depth > 4.0:
                self.detect_publisher.publish(String(data="None"))
                return

            x1, y1, x2, y2 = box.xyxy[0]
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            point_3d_camera = self.unproject_pixel_to_3d(center_x, center_y, depth, intrinsics)
            point_3d_base = self.transform_to_base_frame(point_3d_camera, extrinsics)
            point_msg = Point(x=point_3d_base[0], y=point_3d_base[1], z=point_3d_base[2])
            self.detection_3d_pub.publish(point_msg)
            self.detect_publisher.publish(String(data=camera))
            self.object_position = point_msg

            # Save RGB and Depth images
            if camera == 'head':
                color_image = self.head_rgb_image
                depth_image = self.head_depth_image
            else:
                color_image = self.torso_rgb_image
                depth_image = self.torso_depth_image

            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(color_image, f'{class_name}: {box.conf.item():.2f}', 
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
            rgb_dir = os.path.join(self.rgb_dir, camera)
            depth_dir = os.path.join(self.depth_dir, camera)
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            rgb_image_path = os.path.join(rgb_dir, f"{self.index}.png")
            depth_image_path = os.path.join(depth_dir, f"{self.index}.png")
            print(rgb_image_path)
            cv2.imwrite(rgb_image_path, color_image)
            cv2.imwrite(depth_image_path, depth_image)

            self.move_towards_object()

        else:
            self.stop_robot()
            non_detected_rgb = os.path.join(self.rgb_dir, "no_detections")
            non_detected_rgb_torso = os.path.join(non_detected_rgb, "torso")
            non_detected_rgb_head = os.path.join(non_detected_rgb, "head")

            non_detected_depth = os.path.join(self.depth_dir, "no_detections")
            non_detected_depth_torso = os.path.join(non_detected_depth, "torso")
            non_detected_depth_head = os.path.join(non_detected_depth, "head")

            os.makedirs(non_detected_rgb_head, exist_ok=True)
            os.makedirs(non_detected_rgb_torso, exist_ok=True)
            os.makedirs(non_detected_depth_head, exist_ok=True)
            os.makedirs(non_detected_depth_torso, exist_ok=True)

            cv2.imwrite(os.path.join(non_detected_rgb_torso, f"{self.index}.png"), self.torso_rgb_image)
            cv2.imwrite(os.path.join(non_detected_rgb_head, f"{self.index}.png"), self.head_rgb_image)

            cv2.imwrite(os.path.join(non_detected_depth_torso, f"{self.index}.png"), self.torso_depth_image)
            cv2.imwrite(os.path.join(non_detected_depth_head, f"{self.index}.png"), self.head_depth_image)

            
        self.index += 1



    def unproject_pixel_to_3d(self, center_x, center_y, depth, intrinsics):
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        X_c = (center_x - cx) * depth / fx
        Y_c = (center_y - cy) * depth / fy
        Z_c = depth
        return np.array([X_c, Y_c, Z_c])

    def transform_to_base_frame(self, point_3d_camera, extrinsics):
        R, t = extrinsics['R'], extrinsics['t']
        return np.dot(R, point_3d_camera) + t



    def move_towards_object(self):
        """Moves robot towards detected object based on 3D position and orientation."""
        if self.object_position is None:
            return

        x = self.object_position.x
        y = self.object_position.y

        if abs(x) > 4:
            return  # Ignore if object is too far
        
        # Update the latest orientations of table and sink before moving
        if self.use_odometry:
            self.get_transformed_positions()
            self.desired_orientation_pub.publish(Float32(data=self.desired_orientation_yaw))
            
            # current_yaw = math.atan2(x, y)

            print("Desired orientation:", self.desired_orientation_yaw)
            # print("Current orientation:", current_yaw)

            if (abs(x) < self.goal_tolerance_x
                and abs(y) < self.goal_tolerance_y
                # and (((y > 0) and (y < self.goal_tolerance_y_left)) or ((y < 0) and (-(y) < self.goal_tolerance_y_right)))
                # and abs(current_yaw - self.desired_orientation_yaw) < self.goal_tolerance_ang_z):
                and abs(self.desired_orientation_yaw) < self.goal_tolerance_ang_z):
                self.stop_robot()
                self.end_visual_servoing()
                return

        # Calculate current orientation (yaw) relative to object position
        
        else:
            # Check tolerances in x, y, and angular z (yaw)
            if (abs(x) < self.goal_tolerance_x
                and abs(y) < self.goal_tolerance_y):
                # and (((y > 0) and (y < self.goal_tolerance_y_left)) or ((y < 0) and (-(y) < self.goal_tolerance_y_right)))):
                self.stop_robot()
                self.end_visual_servoing()
                return

        vel_msg = Twist()
        
        # Set linear velocities
        if abs(x) >= self.goal_tolerance_x:
            if abs(x) < 0.8:
                vel_msg.linear.x = self.linear_speed_factor_x * x * 0.8
            else:
                vel_msg.linear.x = self.linear_speed_factor_x * x
        
        if abs(y) >= self.goal_tolerance_y:
        # if (((y < 0) and (y > self.goal_tolerance_y_left)) or ((y > 0) and (-(y) > self.goal_tolerance_y_right))):
            if abs(y) < 0.2:
                vel_msg.linear.y = self.linear_speed_factor_y * y * 0.8 # Uncomment if lateral movement is required
            else:
                vel_msg.linear.y = self.linear_speed_factor_y * y
            if not self.use_odometry:
                vel_msg.angular.z = self.angular_speed_factor * math.atan2(y, x)
        
        if self.use_odometry:
            # Set angular velocity to reach desired yaw orientation
            angular_difference = self.desired_orientation_yaw #- current_yaw
            if abs(angular_difference) >= self.goal_tolerance_ang_z:
                # if abs(angular_difference) < 0.3:
                #     vel_msg.angular.z = self.angular_speed_factor * angular_difference * 0.8
                # else:
                #     vel_msg.angular.z = self.angular_speed_factor * angular_difference
                vel_msg.angular.z = self.angular_speed_factor * angular_difference
                if (vel_msg.angular.z < self.ang_z_thres and vel_msg.angular.z > 0):
                    vel_msg.angular.z = self.ang_z_thres
                if (vel_msg.angular.z > - self.ang_z_thres and vel_msg.angular.z < 0):
                    vel_msg.angular.z = - self.ang_z_thres

                

        self.cmd_vel_pub.publish(vel_msg)


    def stop_robot(self):
        """Stops the robot by publishing zero velocities."""
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        print("Robot stopped")

    def end_visual_servoing(self):
        """Calls /end_visual_servoing service to signal end of visual servoing."""
        if not self.end_servoing_called:
            req = SetBool.Request()
            req.data = True
            future = self.end_visual_servo_client.call_async(req)
            future.add_done_callback(self.end_visual_servo_response_callback)
            self.end_servoing_called = True
            self.visual_servo_started = False  # Stop further detections and movements

    def end_visual_servo_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Successfully ended visual servoing.")
            else:
                self.get_logger().warn("Failed to end visual servoing.")
        except Exception as e:
            self.get_logger().error(f"Service call to end visual servoing failed: {e}")


def main(args=None):
    rclpy.init(args=args)

    username = os.environ.get("USER")
    output_dir_base = f"/home/{username}/poorvi/data/detection"
    next_traj_dir = get_next_traj_dir(output_dir_base)
    # os.makedirs(next_traj_dir, exist_ok=True)

    node = DualCameraProcessorAndFollower(output_dir=next_traj_dir)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
