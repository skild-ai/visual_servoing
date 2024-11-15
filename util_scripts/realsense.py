# quick check to see if both cameras are connected

import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import time
# context has all of the devices and sensors, and provides some additional functionalities.
realsense_ctx = rs.context()  
connected_devices = []

# get serial numbers of connected devices:
for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(
        rs.camera_info.serial_number)
    connected_devices.append(detected_camera)
print(connected_devices)
