import yt_dlp
import cv2
import pytesseract
import jsonlines
import os
import re
import time
from typing import List, Dict, Any

# --- GLOBAL CONFIGURATION AND CONSTANTS ---

# External System Dependency: You must install Tesseract OCR on your system
# and ensure its path is set if not in your environment variables.
# Example path for Windows (uncomment and adjust if needed):
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# YouTube Search Parameters
SEARCH_TERM = "short NLP conference talks"
MAX_DURATION_SECONDS = 240  # 4 minutes maximum
NUM_VIDEOS_TO_FETCH = 10
OCR_FRAME_RATE_SEC = 5  # Process one frame every 5 seconds for transcription

# Output
OUTPUT_FILE = "talks_transcripts.jsonl"
TEMP_VIDEO_FILE = "temp_video.mp4"

# --- HELPER FUNCTIONS ---
def fetch_video_urls(search_query: str, max_duration_sec: int, limit: int) -> List[Dict[str, Any]]:
    
    ydl_opts = {
        'format': 'best[ext=mp4]', # Ensure an mp4 container for easy CV2 reading
        'outtmpl': TEMP_VIDEO_FILE,
        'quiet': True,
        'no_warnings': True,
        'simulate': True,
        'retries': 3,
    }

    search_term = f"ytsearch80:{search_query}"  # fetch more than needed
    video_list = []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_term, download=False)
            entries = info.get('entries', [])

            for entry in entries:
                duration = entry.get('duration')
                if duration and duration <= max_duration_sec:
                    video_list.append({
                        'url': entry['webpage_url'],
                        'title': entry['title'],
                        'duration': duration
                    })
                if len(video_list) >= limit:
                    break
    except Exception as e:
        print(f"  -> An error occurred during video search: {e}")

    return video_list

def download_frames_and_transcribe(video_info: Dict[str, Any], frame_rate: int) -> List[Dict[str, Any]]:
    """
    Downloads a video, extracts frames at a specified rate, performs OCR, and returns
    a list of timestamped transcript segments.

    Args:
        video_info: Dictionary containing the video's 'url'.
        frame_rate: Number of seconds between extracted frames.

    Returns:
        A list of dictionaries, where each dict has 'timestamp' (seconds) and 'text'.
    """
    video_url = video_info['url']
    video_title = video_info['title']
    print(f"  -> Downloading video: {video_title}")

    # 1. Download Video (Unchanged)
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]', 
        'outtmpl': TEMP_VIDEO_FILE,
        'quiet': True,
        'no_warnings': True,
        'retries': 3,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        if os.path.exists(TEMP_VIDEO_FILE):
            os.remove(TEMP_VIDEO_FILE)
        raise RuntimeError(f"Download failed for {video_title}: {e}")
    
    print("  -> Starting frame extraction and OCR...")

    # 2. Extract Frames and Transcribe (Modified)
    # The return type changes from a single string to a list of timestamped dicts
    timestamped_transcript = []
    unique_text_blocks = set() 

    try:
        cap = cv2.VideoCapture(TEMP_VIDEO_FILE)
        if not cap.isOpened():
            raise IOError("Could not open video file with OpenCV.")
        
        # Get properties needed for accurate timing
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # We process frames based on elapsed time (in milliseconds)
        # We need to set the position to jump to the next sampling point
        
        current_time_ms = 0 # Start time in milliseconds
        frame_rate_ms = frame_rate * 1000 # Sample interval in milliseconds
        
        while cap.isOpened() and current_time_ms <= video_info['duration'] * 1000:
            
            # Set the capture position to the desired time (in milliseconds)
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time_ms)
            
            ret, frame = cap.read()
            if not ret:
                break # End of video or error reading frame
            
            # --- OCR Processing (Unchanged) ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) 
            text = pytesseract.image_to_string(thresh, lang='eng')
            cleaned_text = re.sub(r'\s+', ' ', text).strip()
            
            # --- Timestamping and Storage (Modified) ---
            if cleaned_text and cleaned_text not in unique_text_blocks:
                # Calculate the timestamp in seconds
                timestamp_sec = int(current_time_ms / 1000)
                
                timestamped_transcript.append({
                    "timestamp": timestamp_sec,
                    "text": cleaned_text
                })
                unique_text_blocks.add(cleaned_text)
            
            # Move to the next sampling point
            current_time_ms += frame_rate_ms
            
        cap.release()

    except Exception as e:
        print(f"  -> An error occurred during OCR for {video_title}: {e}")
    finally:
        # 3. Cleanup (Unchanged)
        if os.path.exists(TEMP_VIDEO_FILE):
            os.remove(TEMP_VIDEO_FILE)
            
    # Return the list of timestamped segments
    return timestamped_transcript

def save_transcript_data(data: Dict[str, Any], filename: str):
    """
    Appends the processed video data as a JSONL entry to the specified file.
    (Transcript is now a list of segments)

    Args:
        data: Dictionary containing 'title', 'url', and 'transcript' (which is a list).
        filename: The output JSONL file path.
    """
    print(f"  -> Saving data to {filename}")
    try:
        with jsonlines.open(filename, mode='a') as writer:
            writer.write(data)
    except Exception as e:
        print(f"  -> ERROR: Failed to save data to JSONL file: {e}")

# --- MAIN WORKFLOW ---

def main():
    """
    Coordinates the entire script: fetches URLs, processes each video, and saves transcripts.
    """
    print(f"--- 🤖 NLP Talk Transcription Bot Started ---")
    print(f"Search: '{SEARCH_TERM}' | Max Duration: {MAX_DURATION_SECONDS}s")

    start_time = time.time()
    
    # 1. Initialize/clear the output file
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    # 2. Fetch Video URLs
    video_list = fetch_video_urls(SEARCH_TERM, MAX_DURATION_SECONDS, NUM_VIDEOS_TO_FETCH)

    if not video_list:
        print("\n❌ No suitable videos found. Exiting.")
        return

    print(f"\n✅ Found {len(video_list)} suitable videos. Starting transcription...")

    # 3. Process Each Video
    for i, video in enumerate(video_list):
        video_url = video.get('url')
        video_title = video.get('title')
        
        print(f"\n[Video {i+1}/{len(video_list)}]")
        print(f"  Title: {video_title}")
        
        try:
            # Download video frames and perform OCR
            # transcript is now a list of dicts: [{'timestamp': 0, 'text': '...'}, ...]
            transcript_segments = download_frames_and_transcribe(video, OCR_FRAME_RATE_SEC)
            
            if not transcript_segments:
                print("  -> WARNING: No recognizable text found in video frames. Skipping save.")
                continue

            # Prepare data structure for output
            data_to_save = {
                "title": video_title,
                "url": video_url,
                # Store the list of segments
                "transcript_segments": transcript_segments
            }
            
            # Save the result to the JSONL file
            save_transcript_data(data_to_save, OUTPUT_FILE)
            
        except Exception as e:
            print(f"  -> 🛑 CRITICAL ERROR: An unhandled error occurred for {video_title}. Skipping.")
            print(f"  -> Error details: {e}")
            
    end_time = time.time()
    
    print("\n--- ✅ Script Finished ---")
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds.")
    print(f"Transcripts saved to: **{OUTPUT_FILE}**")

if __name__ == "__main__":
    main()