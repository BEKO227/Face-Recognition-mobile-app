import cv2
import time
import threading
import queue
import numpy as np
import os
from dotenv import load_dotenv
from groq import Groq
import json

# Load environment variables from .env file
load_dotenv()

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def analyze_behavior(description, frame):
    """
    Analyze behavior using Groq AI based on the description.
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a behavior analysis model. Analyze whether the video has some one trying to hide his self from the camera or has a behavior of being a theif"},
                {"role": "user", "content": description}
            ],
            temperature=0.2,  # Lower temperature for deterministic output
            max_tokens=150,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Extract the response and return the analysis
        analysis = completion.choices[0].message.content.strip()
        return analysis
    except Exception as e:
        print(f"Error analyzing behavior with Groq: {e}")
        return "Error in behavior analysis."

def process_frame(frame_queue, result_queue, bg_subtractor):
    """
    Processes video frames to detect motion and send them for behavior analysis.
    """
    while True:
        frame_data = frame_queue.get()
        if frame_data is None:  # Exit condition
            break
        frame, timestamp = frame_data

        # Perform background subtraction to detect motion
        fg_mask = bg_subtractor.apply(frame)

        # Pass the processed frame and mask to the result queue
        result_queue.put((frame, fg_mask, timestamp))
        frame_queue.task_done()

def process_video(video_path):
    """
    Handles video input, processes each frame, and integrates Groq AI for behavior analysis.
    """
    # Open the video source (video file path on the laptop)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Frame processing setup
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = 1 / frame_rate

    # Frame queues for communication between threads
    frame_queue = queue.Queue()
    result_queue = queue.Queue()

    # Background subtractor for motion detection
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Start frame processing in a separate thread
    worker_thread = threading.Thread(target=process_frame, args=(frame_queue, result_queue, bg_subtractor))
    worker_thread.start()

    prev_time = time.time()
    video_finished = False

    while True:
        current_time = time.time()

        # Capture and queue frames at the correct frame rate
        if not video_finished and (current_time - prev_time) >= frame_interval:
            ret, frame = cap.read()
            if not ret:
                video_finished = True
            else:
                frame_queue.put((frame, current_time))
                prev_time = current_time

        # Retrieve results from the processing thread
        if not result_queue.empty():
            frame, fg_mask, timestamp = result_queue.get()

            # Perform behavior analysis using Groq AI
            description = f"Motion detected at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}"
            analysis_result = analyze_behavior(description, frame)
            print(f"Behavior Analysis Result: {analysis_result}")

            # Display the video and motion detection results
            motion_frame = np.dstack((fg_mask, fg_mask, fg_mask))
            cv2.imshow('Original Video', frame)
            cv2.imshow('Motion Detection', motion_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit explicitly
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    frame_queue.put(None)
    worker_thread.join()

if __name__ == "__main__":
    # Example usage: Analyze a video file or live camera stream
    process_video("stranger.mp4")  # Replace with your video file path
