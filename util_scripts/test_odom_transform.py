
# To get the position and orientation wrt table/ sink


# import math
# from tf_transformations import euler_from_quaternion

# # # Define table position and quaternion orientation
# position = (13.687, 9.590, 0.0)
# quaternion = (0.0, 0.0, 0.605, 0.798)

# # Define sink position and quaternion orientation
# # position = (13.632, 9.4703, 0.0)
# # quaternion = (0.0, 0.0, -0.790, 0.613)

# # Convert quaternion to roll, pitch, yaw
# roll, pitch, yaw = euler_from_quaternion(quaternion)


# print(f'x: {position[0]}, y: {position[1]}, z: {position[2]}, roll: {roll}, pitch: {pitch}, yaw: {yaw}') 




import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import TransformStamped
import tf_transformations
import numpy as np

class TransformListenerNode(Node):

    def __init__(self, pose_label="table"):
        super().__init__('transform_listener_node')
        
        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to periodically look up the transform
        self.timer = self.create_timer(0.1, self.timer_callback)  # Adjust frequency as needed

        # Pose label (either "table" or "sink")
        self.pose_label = pose_label

        # Define positions and orientations for table and sink in global_frame_2
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

    def timer_callback(self):
        try:
            # Look up the transform from global_frame_2 to base_link
            timeout_duration = rclpy.duration.Duration(seconds=2.0)
            transform = self.tf_buffer.lookup_transform("odom", "base_link", rclpy.time.Time(), timeout_duration)
            
            # Extract translation and rotation quaternion, compute inverse rotation matrix
            trans = transform.transform.translation
            base_link_translation = np.array([trans.x, trans.y, trans.z])
            rot = transform.transform.rotation
            base_link_quaternion = [rot.x, rot.y, rot.z, rot.w]
            inverse_rotation_matrix = tf_transformations.quaternion_matrix(base_link_quaternion).T[:3, :3]

            # Get the position and orientation based on pose_label (table or sink)
            object_position = self.poses[self.pose_label]["position"]
            object_orientation = self.poses[self.pose_label]["orientation"]

            # Compute object position and orientation wrt base_link
            relative_position = object_position - base_link_translation
            position_in_base_link = inverse_rotation_matrix.dot(relative_position)
            object_roll, object_pitch, object_yaw = tf_transformations.euler_from_quaternion(object_orientation)
            base_link_roll, base_link_pitch, base_link_yaw = tf_transformations.euler_from_quaternion(base_link_quaternion)
            roll_in_base_link = object_roll - base_link_roll
            pitch_in_base_link = object_pitch - base_link_pitch
            yaw_in_base_link = object_yaw - base_link_yaw

            # Print result in the requested format
            # print(f'{self.pose_label.capitalize()} wrt base_link -> x: {position_in_base_link[0]}, y: {position_in_base_link[1]}, z: {position_in_base_link[2]}, roll: {roll_in_base_link}, pitch: {pitch_in_base_link}, yaw: {yaw_in_base_link}')
            # Print result in the requested format with values rounded to 2 decimal places
            print(f"x: {round(position_in_base_link[0], 2)}, y: {round(position_in_base_link[1], 2)}, z: {round(position_in_base_link[2], 2)}, "
                f"roll: {round(roll_in_base_link, 2)}, pitch: {round(pitch_in_base_link, 2)}, yaw: {round(yaw_in_base_link, 2)}")


        except TransformException as ex:
            self.get_logger().info(f"Could not transform global_frame_2 to base_link: {ex}")

def main(args=None):
    rclpy.init(args=args)
    # Choose "table" or "sink" to specify which pose to use
    pose_label = "table"  # Change this to "sink" if needed
    node = TransformListenerNode(pose_label)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
