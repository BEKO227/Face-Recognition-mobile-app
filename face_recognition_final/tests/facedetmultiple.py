import cv2
import numpy as np
import ctypes
import face_recognition
import threading
import queue
import time

# Function to get screen resolution
def get_screen_resolution():
    user32 = ctypes.windll.user32
    width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return width, height

# Paths to the model files
model_file = r"C:\Users\Bakr'\OneDrive\Desktop\New\res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_file = r"C:\Users\Bakr'\OneDrive\Desktop\New\deploy.prototxt"

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
known_face_names = ["Authorized Person 1", "Authorized Person 2","Authorized Person 3"]

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y

def encode_face(image_path):
    """
    Encode the face from a given image.
    :param image_path: Path to the image file
    :return: Encoded face or None if no face is found
    """
    try:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            return face_encodings[0]
        else:
            print(f"No faces detected in image: {image_path}")
            return None
    except Exception as e:
        print(f"Error encoding face from {image_path}: {e}")
        return None

def process_frame(frame_queue, result_queue, known_face_encodings, bg_subtractor):
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
                faces.append((x, y, x1-x, y1-y))

        recognized_faces = []
        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_locations = [(0, w, h, 0)]
            face_encodings = face_recognition.face_encodings(rgb_face_image, face_locations)

            if face_encodings:
                face_encoding = face_encodings[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if matches and len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        if best_match_index < len(known_face_names):  # Validate index
                            recognized_faces.append((x, y, w, h, known_face_names[best_match_index], (0, 255, 0)))
                        else:
                            print(f"Error: best_match_index {best_match_index} is out of bounds for known_face_names.")
                    else:
                        recognized_faces.append((x, y, w, h, "Unrecognized Person", (0, 0, 255)))
                else:
                    recognized_faces.append((x, y, w, h, "No Match Found", (255, 0, 0)))
        result_queue.put((frame, fg_mask, recognized_faces, timestamp))
        frame_queue.task_done()

def main(video_source=0):
    global known_face_encodings, known_face_names
    
    image_paths = [
        r"C:\Users\Bakr'\OneDrive\Desktop\New\habiba2.jpeg",
        r"C:\Users\Bakr'\OneDrive\Desktop\New\rana2.jpeg",
        r"C:\Users\Bakr'\OneDrive\Desktop\New\pic2.jpeg"
    ]

    # Load and encode the known faces
    encodings = []
    for image_path in image_paths:
        encoding = encode_face(image_path)
        if encoding is not None:
            encodings.append(encoding)
    known_face_encodings = encodings

    if len(known_face_encodings) != len(known_face_names):
        print("Error: Mismatch between the number of face encodings and known face names.")
        return

    # Open the video source (webcam or video file)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return

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

    worker_thread = threading.Thread(target=process_frame, args=(frame_queue, result_queue, known_face_encodings, bg_subtractor))
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
            else:
                frame_queue.put((frame, current_time))
                prev_time = current_time

        if not result_queue.empty():
            frame, fg_mask, recognized_faces, _ = result_queue.get()

            motion_frame = np.dstack((fg_mask, fg_mask, fg_mask))
            motion_frame_resized = cv2.resize(motion_frame, (main_video_width, main_video_height))

            for (x, y, w, h, name, color) in recognized_faces:
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            cv2.imshow('Original Video', frame)
            cv2.imshow('Motion Detection', motion_frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit explicitly
            break

    cap.release()
    cv2.destroyAllWindows()
    frame_queue.put(None)
    worker_thread.join()

if __name__ == "__main__":
    main(r"C:\Users\Bakr'\OneDrive\Desktop\New\vid1.mp4")
