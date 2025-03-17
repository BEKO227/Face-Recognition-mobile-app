import os
import smtplib  # Import for sending emails
import cv2
import numpy as np
import ctypes
import face_recognition
import threading
import queue
import time
import uuid
import datetime
from collections import defaultdict
from supabase import create_client, Client
from dotenv import load_dotenv
import urllib.request
from email.message import EmailMessage  # Import for constructing email messages
from email.utils import make_msgid

# Load environment variables from .env file
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Email configuration (Read from .env file)
SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
TO_EMAIL = os.getenv('TO_EMAIL')

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
storage_client = supabase.storage

# Function to get screen resolution
def get_screen_resolution():
    user32 = ctypes.windll.user32
    width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return width, height

# Paths to the model files
model_file = "./res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_file = "./deploy.prototxt"

# Check if model files are available
try:
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
except cv2.error as e:
    print(f"Error: {e}")
    print("Ensure that 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000_fp16.caffemodel' are in the correct directory.")
    exit()

# Global variables for mouse position and face encoding
mouse_x, mouse_y = 0, 0
known_face_encodings = []
known_face_names = []

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y

def encode_face_from_image(image):
    """
    Encode the face from a given image array.

    :param image: Image array (numpy array)
    :return: Encoded face or None if no face is found
    """
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_image)
        if face_encodings:
            return face_encodings[0]
        else:
            print("No faces detected in the image.")
            return None
    except Exception as e:
        print(f"Error encoding face: {e}")
        return None

def get_authorized_users():
    try:
        response = supabase.table('authorized_users').select('*').execute()
        return response.data  # Access the data attribute
    except Exception as e:
        print(f"Error fetching authorized users: {e}")
        return []

def send_email(recognized_name, image_bytes, authorized):
    """
    Send an email notification with the face image attached.

    :param recognized_name: Name of the recognized person
    :param image_bytes: Bytes of the face image
    :param authorized: Boolean indicating if the person is authorized
    """
    # Construct the email subject and body based on authorization status
    status = "Authorized" if authorized else "Unauthorized"
    subject = f"{status} user accessed your home"
    body = f"{status} user '{recognized_name}' accessed your home."

    # Create an EmailMessage object
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL
    msg.set_content(body)

    # Add the image as an attachment
    image_cid = make_msgid(domain='xyz.com')
    msg.add_attachment(image_bytes, maintype='image', subtype='jpeg', filename='face.jpg', cid=image_cid)

    # Send the email using SMTP
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            print(f"Email sent to {TO_EMAIL} regarding {recognized_name}.")
    except Exception as e:
        print(f"Error sending email: {e}")

def process_frame(frame_queue, result_queue, known_face_encodings, known_face_names, bg_subtractor):
    processed_faces = defaultdict(lambda: 0)
    COOLDOWN_TIME = 20  # seconds

    while True:
        frame_data = frame_queue.get()
        if frame_data is None:
            break
        frame, timestamp = frame_data

        # Perform background subtraction to detect motion
        fg_mask = bg_subtractor.apply(frame)

        # Perform DNN face detection on the current frame
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x, y, x1, y1) = box.astype("int")
                faces.append((x, y, x1 - x, y1 - y))

        recognized_faces = []
        for (x, y, w_f, h_f) in faces:
            face_image = frame[y:y + h_f, x:x + w_f]
            rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_locations = [(0, w_f, h_f, 0)]
            face_encodings = face_recognition.face_encodings(rgb_face_image, face_locations)

            if face_encodings:
                face_encoding = face_encodings[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        recognized_name = known_face_names[best_match_index]
                        color = (0, 255, 0)
                        authorized = True
                    else:
                        recognized_name = "Unrecognized Person"
                        color = (0, 0, 255)
                        authorized = False
                else:
                    recognized_name = "No Match Found"
                    color = (255, 0, 0)
                    authorized = False

                recognized_faces.append((x, y, w_f, h_f, recognized_name, color))

                current_time = time.time()
                face_key = recognized_name

                # Check if the face has been processed recently
                if current_time - processed_faces[face_key] < COOLDOWN_TIME:
                    print(f"Skipping {recognized_name}, processed {current_time - processed_faces[face_key]:.2f} seconds ago")
                    continue  # Skip processing this face

                # Update the last processed time
                processed_faces[face_key] = current_time

                # Save the face image, upload to Supabase Storage, and insert record into Supabase
                try:
                    # Encode face_image to bytes
                    retval, buffer = cv2.imencode('.jpg', face_image)
                    image_bytes = buffer.tobytes()

                    # Generate unique filename
                    filename = f"{recognized_name}_{uuid.uuid4()}.jpg"

                    # Upload to Supabase Storage under 'authorized_faces' bucket
                    try:
                        storage_response = storage_client.from_('authorized_faces').upload(filename, image_bytes)
                        print(f"Image uploaded successfully: {filename}")
                    except Exception as e:
                        print(f"Error uploading image to storage: {e}")
                        continue

                    # Get public URL
                    try:
                        public_url = storage_client.from_('authorized_faces').get_public_url(filename)
                        if not public_url:
                            print(f"Error getting public URL for image {filename}")
                            continue
                        print(f"Public URL obtained: {public_url}")
                    except Exception as e:
                        print(f"Error getting public URL for image {filename}: {e}")
                        continue

                    # Insert record into Supabase table
                    timestamp_str = datetime.datetime.now().isoformat()
                    try:
                        data = {
                            'name': recognized_name,
                            'timestamp': timestamp_str,
                            'image_url': public_url,
                            'authorized': authorized
                        }
                        db_response = supabase.table('face_logs').insert(data).execute()
                        print(f"Database response: {db_response}")

                        if hasattr(db_response, 'status_code') and db_response.status_code != 201:
                            print(f"Error inserting record into database: {db_response.status_code} {db_response.data}")
                        else:
                            print(f"Record inserted successfully for {recognized_name}")
                    except Exception as e:
                        print(f"Error inserting record into database: {e}")

                    # Send email notification
                    send_email(recognized_name, image_bytes, authorized)

                except Exception as e:
                    print(f"Exception occurred while uploading image and inserting record: {e}")
                    continue

        result_queue.put((frame, fg_mask, recognized_faces, timestamp))
        frame_queue.task_done()

def process_video(video_source, known_face_encodings, known_face_names):
    try:
        # Open the video source (webcam or video file)
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {video_source}")

        # Get video frame rate
        frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = 1 / frame_rate

        # Get screen resolution
        screen_width, screen_height = get_screen_resolution()

        # Frame size for display windows
        main_video_width = int(screen_width * 0.45)
        main_video_height = int(screen_height * 0.7)

        # Create windows for displaying video and motion detection
        cv2.namedWindow('Original Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original Video', main_video_width, main_video_height)
        cv2.setMouseCallback('Original Video', mouse_callback)

        cv2.namedWindow('Motion Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Motion Detection', main_video_width, main_video_height)

        frame_queue = queue.Queue()
        result_queue = queue.Queue()

        # Background subtractor for motion detection
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()

        worker_thread = threading.Thread(
            target=process_frame,
            args=(frame_queue, result_queue, known_face_encodings, known_face_names, bg_subtractor)
        )
        worker_thread.start()

        paused = False
        video_finished = False
        prev_time = time.time()

        while True:
            current_time = time.time()

            if not paused and not video_finished and (current_time - prev_time) >= frame_interval:
                ret, frame = cap.read()
                if not ret:
                    video_finished = True
                    print(f"End of video: {video_source}")
                else:
                    frame_queue.put((frame, current_time))
                    prev_time = current_time

            if not result_queue.empty():
                frame, fg_mask, recognized_faces, _ = result_queue.get()

                motion_frame = np.dstack((fg_mask, fg_mask, fg_mask))
                motion_frame_resized = cv2.resize(motion_frame, (main_video_width, main_video_height))

                for (x, y, w_f, h_f, name, color) in recognized_faces:
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.rectangle(frame, (x, y), (x + w_f, y + h_f), color, 2)

                cv2.imshow('Original Video', frame)
                cv2.imshow('Motion Detection', motion_frame_resized)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit explicitly
                break

    except IOError as e:
        print(f"IOError: {e}")
    except cv2.error as e:
        print(f"OpenCV Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing the video: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        if 'frame_queue' in locals():
            frame_queue.put(None)
        if 'worker_thread' in locals():
            worker_thread.join()


def main(video_sources):
    global known_face_encodings, known_face_names

    # Fetch authorized users from Supabase
    authorized_users = get_authorized_users()

    known_face_encodings = []
    known_face_names = []

    for user in authorized_users:
        name = user['name']
        image_path = user['image_url']  # Assuming this is the path in Supabase Storage

        # Construct the public URL to the image
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/{image_path}"

        # Download the image
        try:
            with urllib.request.urlopen(image_url) as url_response:
                image_data = url_response.read()
                image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                # Encode face from image
                encoding = encode_face_from_image(image)
                if encoding is not None:
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
                else:
                    print(f"No face found in image for user {name}")

        except Exception as e:
            print(f"Error downloading or processing image for {name}: {e}")
            continue

    if len(known_face_encodings) != len(known_face_names):
        print("Error: Mismatch between the number of face encodings and known face names.")
        return

    # Process each video source
    for video_source in video_sources:
        print(f"Processing video: {video_source}")
        process_video(video_source, known_face_encodings, known_face_names)
        # Add a small delay between videos
        time.sleep(1)

if __name__ == "__main__":
    # List of video files to process
    video_files = [r"./hashish2.mp4"]
    main(video_files)