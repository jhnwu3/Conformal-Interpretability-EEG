#!/bin/bash

script_name="conformal_eval_tuev.py"
pids=$(pgrep -f "$script_name")

if [ -z "$pids" ]; then
    echo "No processes with the name '$script_name' found."
else
    for pid in $pids; do
        echo "Terminating process with PID: $pid..."
        kill "$pid"
    done
    echo "All processes with the name '$script_name' have been terminated."
fi
