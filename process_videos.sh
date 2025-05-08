#!/bin/bash

# Get total number of videos for progress reporting
TOTAL_VIDEOS=$(find dataset/train -name "*.mp4" | wc -l)

# Function to process a single video
process_video() {
    local video_path="$1"
    local current=$2
    local total=$3
    
    echo "[$current/$total] Processing $(basename "$video_path")"
    python inference.py -d "$video_path"
}

# Export the function and variables so they can be used by parallel
export -f process_video
export TOTAL_VIDEOS

# Process all videos in parallel with progress bar
find dataset/train -name "*.mp4" | \
    parallel --bar -j 16 process_video {} {#} $TOTAL_VIDEOS