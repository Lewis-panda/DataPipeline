#!/bin/bash

# Define the path to the virtual environment
ENV_PATH="/Users/lewis611036/DataAugmentation/DAenv/bin/activate"
WORK_DIR="/Users/lewis611036/DataAugmentation/Run_Models"

# Remove any existing tmux session named "mysession"
tmux kill-session -t mysession 2>/dev/null

# Create an AppleScript to open a new terminal window and run tmux commands
osascript <<EOF
tell application "Terminal"
    do script "
    cd $WORK_DIR
    tmux new-session -d -s mysession -n run_command
    tmux send-keys -t mysession 'source $ENV_PATH; python3 run_command.py' C-m

    # Split the screen and run run_gemma2.py
    tmux split-window -h -t mysession:0
    tmux send-keys -t mysession 'source $ENV_PATH; python3 run_gemma2.py' C-m

    # Split the screen and run run_llama.py
    tmux split-window -v -t mysession:0.0
    tmux send-keys -t mysession 'source $ENV_PATH; python3 run_llama.py' C-m

    # Split the screen and run run_qwen.py
    tmux split-window -v -t mysession:0.1
    tmux send-keys -t mysession 'source $ENV_PATH; python3 run_qwen.py' C-m

    # Split the screen and run run_yi.py
    tmux split-window -h -t mysession:0.2
    tmux send-keys -t mysession 'source $ENV_PATH; python3 run_yi.py' C-m

    # Split the screen and run run_deepseek.py
    tmux split-window -h -t mysession:0.3
    tmux send-keys -t mysession 'source $ENV_PATH; python3 run_deepseekk.py' C-m

    # Ensure all panes are evenly arranged
    tmux select-layout -t mysession tiled
    tmux -2 attach-session -t mysession
    "
    # Maximize the terminal window
#    set bounds of front window to {0, 0, 1440, 900}
    # Set the terminal window to full screen
    tell application "System Events"
        keystroke "f" using {control down, command down}
    end tell
    
end tell
EOF
