import cv2
import numpy as np
import time
from supabase import create_client, Client
import os
from datetime import datetime

# Initialize Supabase client
supabase: Client = create_client(
    'https://jwnalbtqbzedqrgjtifv.supabase.co',
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp3bmFsYnRxYnplZHFyZ2p0aWZ2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzU4NDgxODcsImV4cCI6MjA1MTQyNDE4N30.CZyUXvdSgQtQG2vh_FuMaWYsaNUQYrFVWDsurRNITz4'
)

class SecuritySystem:
    def __init__(self):
        self.is_armed = False
        self.stream_url = "http://192.168.1.12:8080/video"
        self.cap = cv2.VideoCapture(self.stream_url)
        self.motion_detected = False
        self.motion_counter = 0
        
        # Create a detections directory if it doesn't exist
        if not os.path.exists('detections'):
            os.makedirs('detections')

    def arm_system(self):
        self.is_armed = True
        print("System armed")

    def disarm_system(self):
        self.is_armed = False
        print("System disarmed")

    async def process_detection(self, frame, is_verified: bool):
        try:
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"detections/detection_{timestamp}.jpg"
            
            # Save frame locally
            cv2.imwrite(image_path, frame)
            
            # Upload to Supabase storage
            with open(image_path, 'rb') as f:
                storage_path = f"security-images/detection_{timestamp}.jpg"
                supabase.storage.from_('security-images').upload(storage_path, f)

            # Create security event
            await supabase.table('security_events').insert({
                'image_url': storage_path,
                'is_verified': is_verified,
                'timestamp': time.time(),
                'detection_type': 'motion'
            }).execute()

            # Clean up local file after upload
            os.remove(image_path)
            
            print(f"Detection processed and uploaded: {storage_path}")
            
        except Exception as e:
            print(f"Error processing detection: {str(e)}")

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open video stream")
            return

        # Initialize frame comparison
        ret, frame1 = self.cap.read()
        ret, frame2 = self.cap.read()

        if not ret:
            print("Error: Could not read initial frames")
            return

        print("Motion detection started. Press 'q' to exit.")

        try:
            while self.cap.isOpened():
                if not self.is_armed:
                    time.sleep(1)  # Reduce CPU usage when disarmed
                    continue

                # Compute the absolute difference between frames
                diff = cv2.absdiff(frame1, frame2)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
                dilated = cv2.dilate(thresh, None, iterations=3)
                contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                motion_found = False

                # Process contours and detect significant movement
                for contour in contours:
                    if cv2.contourArea(contour) > 1000:  # Adjust this threshold as needed
                        motion_found = True
                        (x, y, w, h) = cv2.boundingRect(contour)
                        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        if not self.motion_detected:
                            self.motion_detected = True
                            self.motion_counter += 1
                            print(f"Motion detected! Count: {self.motion_counter}")
                            
                            # Process the detection
                            await self.process_detection(frame1, False)  # False for unverified detection

                # Reset motion detection flag if no motion is found
                if not motion_found and self.motion_detected:
                    self.motion_detected = False
                    print("Motion stopped")

                # Display the frame (optional - remove in production)
                cv2.imshow("Security Feed", frame1)

                # Update frames
                frame1 = frame2
                ret, frame2 = self.cap.read()

                if not ret:
                    print("Error: Could not read frame")
                    break

                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error in motion detection: {str(e)}")

        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    security_system = SecuritySystem()
    
    try:
        # Arm the system by default (you can modify this based on your needs)
        security_system.arm_system()
        
        # Run the system
        security_system.run()
        
    except KeyboardInterrupt:
        print("\nSystem shutdown initiated by user")
    finally:
        security_system.cleanup()

if __name__ == "__main__":
    main()