Commands to run for servoing independently
1. bash run_tmux_detection.sh
2. ros2 service call /start_visual_servo std_srvs/srv/SetBool “{data: true}”
3. ros2 topic pub --once /visual_servo_category std_msgs/msg/String “{data: ‘plate’}” # the categories can be plate, spoon, cup
4. python3 end_visual_servoing.py

Notes: https://docs.google.com/document/d/149MpQcg0I83XUtM3vTYfUpEGR6bURg2INi_Xe0oH81w/edit?usp=sharing
Fintune runs details (models in poorvi@skild7.local:/home/poorvi/h1_detection/yolov10/runs/detect): 
https://docs.google.com/spreadsheets/d/1gc1beRZHL4GFyTsNoTo6j-ubfdWYQOYovCdqTd9dhv4/edit?usp=sharing
