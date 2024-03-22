#!/bin/bash
# This script follows the log files for jobs with a specific name.
# Usage: ./view_rolling_logs.sh job_name

# Find job IDs for jobs with the given name
job_ids=$(squeue --name="$1" --format=%A --noheader)

# Loop through each job ID and follow its log file
for id in $job_ids; do
    # Modify the log file pattern as per your setup
    log_file="./${1}-${id}.out"
    
    # Check if log file exists and then follow it
    if [ -f "$log_file" ]; then
        echo "Following log for Job ID: $id"
        tail -f "$log_file" &
    else
        echo "Log file for Job ID $id does not exist yet."
    fi
done

# Wait for tail processes to finish if you want to keep the script running
wait
