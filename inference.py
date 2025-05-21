# coding: utf-8

"""
The entrance of humans
"""

import os
import os.path as osp
import tyro
import subprocess
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")


def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
        )

    fast_check_args(args)

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)


    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    # run with optional original_video parameter
    original_video_path = args.original_video if hasattr(args, 'original_video') and args.original_video else "animations/original.mp4"
    
    # Derive original video path from driving path, if applicable
    if args.driving and args.driving.endswith("_reconstructed.pkl"):
        original_video_path = args.driving.replace("_reconstructed.pkl", ".mp4")
        original_video_path = original_video_path.replace("dataset/pickles/", "dataset/train/")
        
        # Get video_id from driving path
        video_id = os.path.basename(args.driving).replace("_reconstructed.pkl", "")
        
        # Check for baseline video
        baseline_video_path = os.path.join(args.output_dir, f"{video_id}_baseline.mp4")
        
        if not os.path.exists(baseline_video_path):
            print(f"Baseline video not found: {baseline_video_path}")
            
            # Try to create baseline video using original pickle
            original_pickle = f"dataset/pickles/{video_id}.pkl"
            if os.path.exists(original_pickle):
                print(f"Found original pickle: {original_pickle}")
                
                # Save current driving path
                temp_driving = args.driving
                
                # Set driving to original pickle and run pipeline
                args.driving = original_pickle
                print(f"Running pipeline with original pickle to generate baseline...")
                
                # This will create the baseline video
                live_portrait_pipeline.execute(args, original_video_path)
                
                # Restore driving path
                args.driving = temp_driving
                
                # Now the baseline should exist for the second run
                if os.path.exists(baseline_video_path):
                    print(f"Successfully created baseline: {baseline_video_path}")
                else:
                    print("Failed to create baseline video")
            else:
                print(f"Original pickle not found: {original_pickle}")
        else:
            print(f"Found baseline video: {baseline_video_path}")
    
    # Find the next available index for the output video
    if args.driving and args.driving.endswith("_reconstructed.pkl"):
        video_id = os.path.basename(args.driving).replace("_reconstructed.pkl", "")
        index = 1
        while True:
            test_path = os.path.join(args.output_dir, f"{video_id}_{index}.mp4")
            if not os.path.exists(test_path):
                break
            index += 1
        
        # Temporarily save the output index
        args.output_index = index
        print(f"Output video will be named: {video_id}_{index}.mp4")
    
    # Check if original video exists and report
    if os.path.exists(original_video_path):
        print(f"Using original video: {original_video_path}")
    else:
        print(f"Original video not found: {original_video_path}")
        
    # Run the pipeline (with baseline if available)
    baseline_video_path = os.path.join(args.output_dir, f"{video_id}_baseline.mp4") if args.driving and args.driving.endswith("_reconstructed.pkl") else None
    if baseline_video_path and os.path.exists(baseline_video_path):
        print(f"Using baseline video: {baseline_video_path}")
    else:
        baseline_video_path = None
        
    live_portrait_pipeline.execute(args, original_video_path, baseline_video_path)


if __name__ == "__main__":
    main()
