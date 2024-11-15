# quick check to see if odometry gets updated reliably
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import TransformStamped
import tf_transformations

class TransformListenerNode(Node):

    def __init__(self):
        super().__init__('transform_listener_node')
        
        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to periodically look up the transform
        self.timer = self.create_timer(1.0, self.timer_callback)  # Adjust frequency as needed

    def timer_callback(self):
        try:
            # Look up the transform from /odom to /base_link
            timeout_duration = rclpy.duration.Duration(seconds=2.0)
            t = self.tf_buffer.lookup_transform("global_frame_2", "base_link", rclpy.time.Time(), timeout_duration)
            
            # Extract translation
            x = round(t.transform.translation.x, 2)
            y = round(t.transform.translation.y, 2)
            z = round(t.transform.translation.z, 2)

            # Extract rotation quaternion
            qx = t.transform.rotation.x
            qy = t.transform.rotation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w

            # Convert quaternion to roll, pitch, yaw
            roll, pitch, yaw = tf_transformations.euler_from_quaternion([qx, qy, qz, qw])
            roll = round(roll, 2)
            pitch = round(pitch, 2)
            yaw = round(yaw, 2)

            # Print the x, y, z, roll, pitch, yaw
            print(f'x: {x}, y: {y}, z: {z}, roll: {roll}, pitch: {pitch}, yaw: {yaw}') 
            # this is the current position of robot wrt global_frame_2
        
        except TransformException as ex:
            self.get_logger().info(f"Could not transform /global_frame_2 to /base_link: {ex}")

def main(args=None):
    rclpy.init(args=args)
    node = TransformListenerNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
