import sys
import numpy as np
from dynamixel.active_cam import DynamixelAgent

class SkildHead():
    def __init__(self, dynamixel_port='/dev/ttyUSB0', dynamixel_name='head-1', head_limits=[[-1.308958333, 1.308958333], [-0.349055556, 1.5708]], version='v1'):
        self.dynamixel_port = dynamixel_port
        self.dynamixel_name = dynamixel_name
        self.head_limits = head_limits
        self.version = version
        
        self.load_motors()

    def compute_head_commands(self, yaw, pitch):
        if self.version == 'v1':
            commands = [yaw + pitch, yaw - pitch]
        else:
            commands = [yaw, pitch]
        return commands
    
    def command_head(self, yaw, pitch):
        # Clip yaw and pitch to within the head limits
        clipped_yaw = np.clip(yaw, self.head_limits[0][0], self.head_limits[0][1])
        clipped_pitch = np.clip(pitch, self.head_limits[1][0], self.head_limits[1][1])
        
        # Send commands to the motors
        self.agent._robot.command_joint_state(self.compute_head_commands(clipped_yaw, clipped_pitch))
    
    def load_motors(self):
        # Initialize the Dynamixel agent
        self.agent = DynamixelAgent(port=self.dynamixel_port, name=self.dynamixel_name)
        self.agent._robot.set_torque_mode(True)

    def close(self):
        # Disable torque mode on exit
        self.agent._robot.set_torque_mode(False)
        self.agent._robot.close()

if __name__ == '__main__':
    head = SkildHead(
        dynamixel_port='/dev/ttyUSB0',
        dynamixel_name='head-2',
        version='v2'
    )
    
    # Example command: set yaw to 0 and pitch to 0.785 radians
    head.command_head(0.0, 0.3)
