"""
Extract frames from a video and create a FiftyOne image dataset.

This script takes a video file, extracts all frames, and creates a FiftyOne
image dataset with metadata including frame_number and video_timestamp.
"""

import cv2
import fiftyone as fo
import numpy as np
from pathlib import Path
import tempfile
import os


def extract_frames_to_dataset(
    video_path: str,
    dataset_name: str = "basketball_frames",
    output_dir: str = None,
    frame_skip: int = 1,
):
    """
    Extract frames from a video and create a FiftyOne image dataset.
    
    Args:
        video_path: Path to the input video file
        dataset_name: Name for the FiftyOne dataset
        output_dir: Directory to save frame images (optional, uses temp dir if not provided)
        frame_skip: Extract every Nth frame (1 = all frames, 2 = every other frame, etc.)
    
    Returns:
        The created FiftyOne dataset
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frame skip: {frame_skip}")
    
    # Create output directory for frames
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="frames_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    output_dir = Path(output_dir)
    print(f"  Output directory: {output_dir}")
    
    dataset = fo.Dataset(dataset_name, overwrite=True)
    
    # Extract frames
    frame_number = 0
    extracted_count = 0
    
    print("\nExtracting frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only extract frames based on frame_skip
        if frame_number % frame_skip == 0:
            # Calculate timestamp in seconds
            video_timestamp = frame_number / fps if fps > 0 else 0.0
            
            # Save frame as image
            frame_filename = f"frame_{frame_number:06d}.jpg"
            frame_path = output_dir / frame_filename
            
            cv2.imwrite(str(frame_path), frame)
            
            # Create sample with metadata
            sample = fo.Sample(
                filepath=str(frame_path),
                frame_number=frame_number,
                video_timestamp=video_timestamp,
            )
            
            dataset.add_sample(sample)
            extracted_count += 1
            
            if extracted_count % 100 == 0:
                print(f"  Extracted {extracted_count} frames...")
        
        frame_number += 1
    
    cap.release()
    
    print(f"\nExtraction complete!")
    print(f"  Total frames extracted: {extracted_count}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Dataset size: {len(dataset)} samples")
    
    return dataset


if __name__ == "__main__":
    # Path to the stitched video
    video_path = Path(__file__).parent.parent / "stitched_basketball_videos.mp4"
    
    # Create dataset
    dataset = extract_frames_to_dataset(
        video_path=str(video_path),
        dataset_name="basketball_frames",
        frame_skip=4,  # Extract all frameframes (set to 2 for every other frame, etc.)
    )
    
    # Print dataset info
    print("\n" + "="*50)
    print("Dataset created successfully!")
    print("="*50)
    print(f"Dataset name: {dataset.name}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Sample fields: {dataset.get_field_schema()}")
    
    # Show a sample
    if len(dataset) > 0:
        sample = dataset.first()
        print(f"\nFirst sample:")
        print(f"  Filepath: {sample.filepath}")
        print(f"  Frame number: {sample.frame_number}")
        print(f"  Video timestamp: {sample.video_timestamp:.3f} seconds")
