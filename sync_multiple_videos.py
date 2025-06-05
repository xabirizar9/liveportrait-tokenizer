#!/usr/bin/env python3
"""Sync multiple videos into a single overlaid video."""

import subprocess
import sys
from pathlib import Path


VIDEO_PATHS = [
    "animations/female_24fps_reconstructed_1_concat.mp4",
    "animations/WQvT1_tQDhg_22_reconstructed_1.mp4",
    "animations/WQvT1_tQDhg_22_reconstructed_1_resampled.mp4"
]


def run_ffmpeg(cmd):
    """Run ffmpeg command and handle errors."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)


def get_duration(video_path):
    """Get video duration in seconds."""
    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
           '-of', 'csv=p=0', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def sync_videos(video_paths, output='output.mp4'):
    """Sync videos side by side or in grid."""
    n = len(video_paths)
    
    if n == 1:
        # Single video with title
        name = Path(video_paths[0]).stem
        filter_complex = f"[0:v]drawtext=text='{name}':fontsize=24:fontcolor=white:x=10:y=10[v]"
        
    elif n == 2:
        # Side by side
        names = [Path(p).stem for p in video_paths]
        filter_complex = (
            f"[0:v]scale=512:512,drawtext=text='{names[0]}':fontsize=20:fontcolor=white:x=10:y=10[v0];"
            f"[1:v]scale=512:512,drawtext=text='{names[1]}':fontsize=20:fontcolor=white:x=10:y=10[v1];"
            f"[v0][v1]hstack[v]"
        )
        
    else:
        # 2x2 grid (up to 4 videos)
        names = [Path(p).stem for p in video_paths[:4]]
        filter_parts = []
        
        for i, name in enumerate(names):
            filter_parts.append(f"[{i}:v]scale=512:512,drawtext=text='{name}':fontsize=16:fontcolor=white:x=10:y=10[v{i}]")
        
        if n == 3:
            filter_parts.append("color=black:s=512x512:d=10[black]")
            filter_parts.append("[v0][v1]hstack[top]")
            filter_parts.append("[v2][black]hstack[bottom]")
        else:
            filter_parts.append("[v0][v1]hstack[top]")
            filter_parts.append("[v2][v3]hstack[bottom]")
        
        filter_parts.append("[top][bottom]vstack[v]")
        filter_complex = ";".join(filter_parts)
    
    # Get shortest duration
    durations = [get_duration(p) for p in video_paths]
    min_duration = min(durations)
    
    # Build and run ffmpeg command
    cmd = ['ffmpeg', '-y']
    for path in video_paths:
        cmd.extend(['-i', str(path)])
    
    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-t', str(min_duration),
        '-c:v', 'libx264',
        '-preset', 'fast',
        str(output)
    ])
    
    run_ffmpeg(cmd)
    print(f"Created: {output}")


def main():
    # Check dependencies
    for tool in ['ffmpeg', 'ffprobe']:
        if subprocess.run(['which', tool], capture_output=True).returncode != 0:
            print(f"Error: {tool} not found", file=sys.stderr)
            sys.exit(1)
    
    # Find existing videos
    videos = [p for p in VIDEO_PATHS if Path(p).exists()]
    
    if not videos:
        print("Error: No videos found", file=sys.stderr)
        sys.exit(1)
    
    if len(videos) > 4:
        videos = videos[:4]
        print("Warning: Using only first 4 videos")
    
    sync_videos(videos)


if __name__ == "__main__":
    main()
