import cv2
import numpy as np
import ctypes
import face_recognition
import threading
import queue
import time

def get_screen_resolution():
    """Get the screen resolution."""
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def encode_face(image_path):
    """Encode the face from an image."""
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            return encodings[0]
        else:
            print(f"No face found in image: {image_path}")
    except Exception as e:
        print(f"Error encoding face: {e}")
    return None

def process_frame(frame_queue, result_queue, known_encodings, bg_subtractor):
    """Process frames for motion detection and face recognition."""
    while True:
        frame_data = frame_queue.get()
        if frame_data is None:
            break
        frame, timestamp = frame_data

        # Perform motion detection
        fg_mask = bg_subtractor.apply(frame)

        # Face detection using DNN
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                faces.append((x, y, x1 - x, y1 - y))

        recognized_faces = []
        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_locations = [(0, w, h, 0)]
            encodings = face_recognition.face_encodings(rgb_face, face_locations)

            if encodings:
                match = face_recognition.compare_faces(known_encodings, encodings[0])
                distances = face_recognition.face_distance(known_encodings, encodings[0])
                if match[np.argmin(distances)]:
                    recognized_faces.append((x, y, w, h, "Authorized", (0, 255, 0)))
                else:
                    recognized_faces.append((x, y, w, h, "Unknown", (0, 0, 255)))
            else:
                recognized_faces.append((x, y, w, h, "Unknown", (0, 0, 255)))

        result_queue.put((frame, fg_mask, recognized_faces, timestamp))
        frame_queue.task_done()

def main(video_source=0):
    """Main function to run motion detection and face recognition."""
    global net

    # Load the face detection model
    model = r"C:\Users\Bakr'\OneDrive\Desktop\New\res10_300x300_ssd_iter_140000_fp16.caffemodel"
    config = r"C:\Users\Bakr'\OneDrive\Desktop\New\deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config, model)

    # Encode known face
    known_face = encode_face(r"C:\Users\Bakr'\OneDrive\Desktop\New\hashish.jpeg")
    if known_face is None:
        print("Error encoding known face. Exiting.")
        return
    known_encodings = [known_face]

    # Open video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    # Retrieve video FPS
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    delay = max(1, int(1000 / fps))

    screen_width, screen_height = get_screen_resolution()
    window_size = (int(screen_width * 0.5), int(screen_height * 0.5))

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", *window_size)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    frame_queue = queue.Queue()
    result_queue = queue.Queue()

    worker_thread = threading.Thread(
        target=process_frame,
        args=(frame_queue, result_queue, known_encodings, bg_subtractor),
        daemon=True,
    )
    worker_thread.start()

    paused = False
    video_finished = False

    while True:
        if not paused and not video_finished:
            ret, frame = cap.read()
            if not ret:
                video_finished = True
                break

            frame_queue.put((frame, time.time()))

        if not result_queue.empty():
            frame, fg_mask, faces, _ = result_queue.get()

            for (x, y, w, h, name, color) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Video", frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    frame_queue.put(None)
    worker_thread.join()

if __name__ == "__main__":
    main(r"C:\Users\Bakr'\OneDrive\Desktop\New\hashish2.mp4")
