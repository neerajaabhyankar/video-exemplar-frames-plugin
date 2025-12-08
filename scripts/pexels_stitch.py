# Create a video dataset from internet videos
import requests
import tempfile
from pathlib import Path
import subprocess
import re
import os

API_KEY = "tRUWNkToqQEtWE90XyNuMuhg7DJnN1lkJtGMpPo5RoY5CfsEp3SZsMiC"
video_url = 'https://www.pexels.com/video/man-playing-basketball-5192069/'

VIDEO_PATHS = [
    "/Users/neeraja/Downloads/basketball/5192069-sd_640_360_30fps.mp4",
    "/Users/neeraja/Downloads/basketball/5192025-sd_640_360_30fps.mp4",
    "/Users/neeraja/Downloads/basketball/5192077-sd_640_360_30fps.mp4",
    "/Users/neeraja/Downloads/basketball/5192149-sd_640_360_30fps.mp4",
    "/Users/neeraja/Downloads/basketball/5192076-sd_640_360_30fps.mp4",
    "/Users/neeraja/Downloads/basketball/5192077-sd_640_360_30fps.mp4",
    "/Users/neeraja/Downloads/basketball/5192072-sd_640_360_30fps.mp4",
    "/Users/neeraja/Downloads/basketball/5192069-sd_640_360_30fps.mp4",
    "/Users/neeraja/Downloads/basketball/5192154-sd_640_360_30fps.mp4",
    "/Users/neeraja/Downloads/basketball/5192074-sd_640_360_30fps.mp4",
]


def extract_video_id(url):
    """Extract video ID from Pexels video URL."""
    # Pexels URLs can be in formats like:
    # https://www.pexels.com/video/man-playing-basketball-5192069/
    # https://www.pexels.com/video/5192069/
    # The ID is always at the end before the trailing slash
    match = re.search(r'-(\d+)/?$', url)
    if match:
        return match.group(1)
    # Fallback: try to match digits at the end if no dash pattern found
    match = re.search(r'/(\d+)/?$', url)
    if match:
        return match.group(1)
    return None


def get_single_video(video_id, api_key, output_dir=None):
    """
    Get a single video from Pexels API by video ID.
    
    Args:
        video_id: Pexels video ID (string or int)
        api_key: Pexels API key
        output_dir: Directory to save the video (optional, uses temp dir if not provided)
    
    Returns:
        Path to downloaded video file
    """
    headers = {'Authorization': api_key}
    url = f'https://api.pexels.com/videos/videos/{video_id}'
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve video: {response.status_code} - {response.text}")
    
    video_data = response.json()
    
    # Get the highest quality video file
    video_files = video_data.get('video_files', [])
    if not video_files:
        raise Exception("No video files found in response")
    
    # Sort by quality/width, get the best one
    best_video = max(video_files, key=lambda x: x.get('width', 0) * x.get('height', 0))
    video_download_url = best_video['link']
    
    # Download the video
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = Path(output_dir) / f"video_{video_id}.mp4"
    
    print(f"Downloading video {video_id}...")
    vid_response = requests.get(video_download_url, stream=True, headers={"User-Agent": ""})
    vid_response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in vid_response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Video saved to: {output_path}")
    return output_path


def stitch_videos(video_paths, output_path):
    """
    Stitch multiple videos together using ffmpeg.
    
    Args:
        video_paths: List of paths to video files
        output_path: Path for the output stitched video
    
    Returns:
        Path to the stitched video
    """
    if not video_paths:
        raise ValueError("No video paths provided")
    
    if len(video_paths) == 1:
        # If only one video, just copy it
        import shutil
        shutil.copy(video_paths[0], output_path)
        return output_path
    
    # Create a temporary file list for ffmpeg concat
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        for video_path in video_paths:
            # Escape single quotes and backslashes for ffmpeg
            escaped_path = str(video_path).replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
    
    try:
        # Use ffmpeg concat demuxer
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            str(output_path)
        ]
        
        print(f"Stitching {len(video_paths)} videos together...")
        result = subprocess.run(
            ffmpeg_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Stitched video saved to: {output_path}")
        return output_path
    finally:
        # Clean up temp file
        os.unlink(concat_file)


# # Get a single video
# if __name__ == "__main__":
#     video_id = extract_video_id(video_url)
#     if video_id:
#         print(f"Extracted video ID: {video_id}")
#         output_dir = tempfile.mkdtemp()
#         try:
#             video_path = get_single_video(video_id, API_KEY, output_dir)
#             print(f"Successfully downloaded video to: {video_path}")
#         except Exception as e:
#             print(f"Error: {e}")
#     else:
#         print("Could not extract video ID from URL")

# # Stitch videos
if __name__ == "__main__":
    video_paths = VIDEO_PATHS
    output_path = "stitched_basketball_videos.mp4"
    stitch_videos(video_paths, output_path)