# #!/bin/bash

# Start a new tmux session named h1_detection
tmux new-session -d -s h1_detection

# Window 0: Camera - remains as it is
tmux new-window -t h1_detection:0 -n Camera
tmux send-keys -t h1_detection:0.0 '1' C-m
tmux send-keys -t h1_detection:0.0 'if conda info --envs | grep "*"; then conda deactivate; fi' C-m
tmux send-keys -t h1_detection:0.0 'cd poorvi' C-m
tmux send-keys -t h1_detection:0.0 'python3 publish_rgbd.py' C-m

# Window 1: Servoing
tmux new-window -t h1_detection:1 -n Servoing
tmux send-keys -t h1_detection:1.0 '1' C-m
tmux send-keys -t h1_detection:1.0 'conda activate yolov10 || echo "Already in yolov10"' C-m
tmux send-keys -t h1_detection:1.0 'cd /home/unitree/poorvi/yolov10' C-m
tmux send-keys -t h1_detection:1.0 'python3 visual_servoing.py' C-m

# Open three additional panes in window 1
tmux split-window -t h1_detection:1
tmux send-keys -t h1_detection:1.1 '1' C-m
tmux send-keys -t h1_detection:1.1 'sleep 20 && ros2 topic echo /detect' C-m

tmux split-window -t h1_detection:1
tmux send-keys -t h1_detection:1.2 '1' C-m
tmux send-keys -t h1_detection:1.2 'sleep 20 && ros2 topic echo /detection_3d' C-m

tmux split-window -t h1_detection:1
tmux send-keys -t h1_detection:1.3 '1' C-m
tmux send-keys -t h1_detection:1.3 'sleep 20 && ros2 topic echo /cmd_vel' C-m

# tmux split-window -t h1_detection:1
# tmux send-keys -t h1_detection:1.4 '1' C-m
# tmux send-keys -t h1_detection:1.4 'sleep 20 && ros2 topic echo /desired_orientation_yaw' C-m


# Attach to the session
tmux attach -t h1_detection


# # ros2 service call start_visual_servo std_srvs/srv/SetBool "{data: true}"
# # ros2 topic pub --once /visual_servo_category std_msgs/msg/String "{data: "spoon"}"