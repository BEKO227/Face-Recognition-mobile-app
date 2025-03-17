import cv2
import numpy as np
import ctypes

# Function to get screen resolution
def get_screen_resolution():
    user32 = ctypes.windll.user32
    width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return width, height

# Paths to the model files
model_file = r"E:\ENGR YEAR 4\Semster 8\ECEN 493 Grad 1\Face recognetion model\Final_models\res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_file = r"E:\ENGR YEAR 4\Semster 8\ECEN 493 Grad 1\Face recognetion model\Final_models\deploy.prototxt"

# Check if model files are available
try:
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
except cv2.error as e:
    print(f"Error: {e}")
    print("Ensure that 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000_fp16.caffemodel' are in the correct directory.")
    exit()

# Global variables for mouse position
mouse_x, mouse_y = 0, 0

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y

def main(video_source=0):
    global mouse_x, mouse_y

    # Open the video source
    cap = cv2.VideoCapture(r"E:\ENGR YEAR 4\Semster 8\ECEN 493 Grad 1\Face recognetion model\test_multi.mp4")
    
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return
    
    # Get screen resolution
    screen_width, screen_height = get_screen_resolution()

    # Calculate the desired frame sizes (45% width and 70% height for main video, same for motion window)
    main_video_width = int(screen_width * 0.45)
    main_video_height = int(screen_height * 0.7)

    # Create a separate window for motion detection
    cv2.namedWindow('Motion Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Motion Detection', main_video_width, main_video_height)

    # Variables for control
    paused = False
    zoom_level = 1.0
    zoom_center = None

    cv2.namedWindow('Original Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Video', main_video_width, main_video_height)
    cv2.setMouseCallback('Original Video', mouse_callback)

    previous_frame = None
    previous_faces = []

    # Background subtractor for motion detection
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

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
                if confidence > 0.3:  # Lowering confidence threshold to 0.3
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    (x, y, x1, y1) = box.astype("int")
                    faces.append((x, y, x1-x, y1-y))

            # Store current frame and faces
            previous_frame = frame
            previous_faces = faces
        else:
            # If paused, retain the previous frame and face detections
            frame = previous_frame
            faces = previous_faces
        
        # Draw a rectangle around the faces in the original video window
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Apply motion mask to frame for motion detection window
        motion_frame = np.dstack((fg_mask, fg_mask, fg_mask))  # Convert single channel mask to 3 channel
        motion_frame_resized = cv2.resize(motion_frame, (main_video_width, main_video_height))

        # Display frames in respective windows
        cv2.imshow('Original Video', frame)
        cv2.imshow('Motion Detection', motion_frame_resized)

        # Wait for key press
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break  # Quit the program
        elif key == ord('p'):
            paused = not paused
        elif key == ord('f'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 30)  # forward 30 frames
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 30)  # rewind 30 frames
        elif key == ord('+'):
            zoom_level = min(zoom_level + 0.1, 3.0)  # zoom in
        elif key == ord('-'):
            zoom_level = max(zoom_level - 0.1, 1.0)  # zoom out
        elif key == ord('z'):
            zoom_center = None  # reset zoom center to default
        elif key == ord('m'):
            # Move the zoom center to the current mouse position
            zoom_center = (mouse_x, mouse_y)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(0)  # 0 indicates the default webcam or video source
    
#Key Functionality:
#'q': Quit the program.
#'p': Pause/resume the video.
#'f': Forward 30 frames (not applicable to real-time video but kept for consistency).
#'r': Rewind 30 frames (not applicable to real-time video but kept for consistency).
#'+': Zoom in.
#'-': Zoom out.
#'z': Reset zoom center to default.
#'m': Move the zoom center to the current mouse position