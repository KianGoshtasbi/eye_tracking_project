import cv2
import mediapipe as mp
import numpy as np

class EyeTracker:
    """
    EyeTracker: Real-time eye landmark detection and Eye Aspect Ratio (EAR) calculation.

    This class uses MediaPipe Face Mesh to detect facial landmarks,
    extracts eye landmarks, computes the Eye Aspect Ratio (EAR), 
    and visualizes the eyes with color-coded contours based on open/closed state.

    Attributes:
        ear_threshold (float): Threshold for EAR to determine eye closure.
        LEFT_EYE (list[int]): Landmark indices for left eye.
        RIGHT_EYE (list[int]): Landmark indices for right eye.
    """
    def __init__(self, ear_threshold=0.21, max_faces=1):
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        
        self.ear_threshold = ear_threshold
    
    #Formula to calculate Eye Aspect Ratio
    def calculate_ear(self, eye_landmarks):
        p1, p2, p3, p4, p5, p6 = eye_landmarks
        ear = (self.euclidean_distance(p2, p6) + self.euclidean_distance(p3, p5)) / (2.0 * self.euclidean_distance(p1, p4))
        return ear
    
    def get_eye_landmarks(self, landmarks, indices, frame_w, frame_h):
        """
        get_eye_landmarks: since results.multi_face_landmarks[0].landmark returns normalized x and y coordinates between 0-1
        relative to the images width and height 
        1. we go through the specified indices for LEFT_EYE and RIGHT_EYE for the landmark
        2. for each indicie we multiply the x value by the width, and y value by the height to get the actual pixel coordinates
        3. cast them to integers to get integer coordinates
        4. put them in a tuple and make a list of coordinates to return

        ARGS:
            landmarks (list): specific key points on a person's face (list of x and y values of each point)
            indices (list): specific locations of the landmark we want to extract
            frame_w (int): width of the frame
            frame_h(int): height of the frame
        """
        coords = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in indices]
        return coords
    
    #use np.linalg.norm for efficiency
    def euclidean_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def process_frame(self, frame):
        """
        Process a single video frame for real-time eye detection and EAR calculation.

        Steps:
            1. Get the frame dimensions (height and width).
            2. Convert the frame from BGR to RGB .
            3. Use MediaPipe Face Mesh to detect facial landmarks.
            4. If a face is detected:
                a. Extract the left and right eye landmarks using `get_eye_landmarks`.
                b. Compute the Eye Aspect Ratio (EAR) for each eye using `calculate_ear`.
                c. Calculate the average EAR across both eyes.
                d. Determine the eye state (OPEN or CLOSED) based on EAR threshold.
                e. Choose a color: green if eyes are open, red if closed.
                f. Draw lines around each eye using OpenCV.
                g. Display EAR value and eye state as text on the frame.
            5. If no face is detected, display a warning text on the frame.

        Args:
            frame (np.ndarray): A single frame captured from the webcam (BGR format).

        Returns:
            np.ndarray: The annotated frame with eye contours and EAR/eye state displayed.
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_coords = self.get_eye_landmarks(landmarks, self.LEFT_EYE, w, h)
            right_coords = self.get_eye_landmarks(landmarks, self.RIGHT_EYE, w, h)
            
            left_ear = self.calculate_ear(left_coords)
            right_ear = self.calculate_ear(right_coords)
            avg_ear = (left_ear + right_ear) / 2.0
            
            
            color = (0, 255, 0) if avg_ear >= self.ear_threshold else (0, 0, 255)
            eye_state = "OPEN" if avg_ear >= self.ear_threshold else "CLOSED"
            print(f"[INFO] EAR: {avg_ear:.2f}, Eyes {eye_state}")
            
            #Draws lines around the eyes (Double ensure they are integers with np.int32)
            cv2.polylines(frame, [np.array(left_coords, np.int32)], True, color, 2)
            cv2.polylines(frame, [np.array(right_coords, np.int32)], True, color, 2)
            
            #Displays whether eyes are open or closed along with EAR value
            cv2.putText(frame, f"EAR: {avg_ear:.2f} - {eye_state}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            print("[WARNING] No face detected")
            cv2.putText(frame, "No face detected", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame
    
    def run(self):
        """
        Main loop to capture webcam frames, process them, and display annotated output in real-time.

        Steps:
            1. Open the default webcam using OpenCV VideoCapture.
            2. Check if the webcam is successfully opened; if not, print an error and exit.
            3. Start an infinite loop:
                a. Capture a frame from the webcam.
                b. If frame capture fails, print an error and exit the loop.
                c. Process the frame using `process_frame` to annotate eyes and EAR.
                d. Display the annotated frame in a window named "EyeTracker".
                e. Check if the 'q' key is pressed; if so, exit the loop.
            4. After exiting the loop, release the webcam resource.
            5. Close all OpenCV display windows to free resources.

        Args:
            None

        Returns:
            None
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: webcam will not open")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            annotated_frame = self.process_frame(frame)
            cv2.imshow("EyeTracker", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    tracker = EyeTracker()
    tracker.run()



