#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

class EndVisualServoingService(Node):
    def __init__(self):
        super().__init__('end_visual_servoing_service')
        self.service = self.create_service(SetBool, 'end_visual_servo', self.end_visual_servoing_callback)
        self.get_logger().info("Service /end_visual_servoing is ready.")

    def end_visual_servoing_callback(self, request, response):
        if request.data:
            self.get_logger().info("End visual servoing request received.")
            response.success = True
            response.message = "Visual servoing ended successfully."
        else:
            response.success = False
            response.message = "Visual servoing not ended."
        return response

def main(args=None):
    rclpy.init(args=args)
    node = EndVisualServoingService()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
