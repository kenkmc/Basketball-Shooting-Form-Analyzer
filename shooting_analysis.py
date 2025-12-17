import cv2
import mediapipe as mp
import numpy as np
import math
import csv
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import threading
import os

class BasketballShotAnalyzer:
    def __init__(self, video_source=0, shooter_height_cm=180, output_path=None):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.video_source = video_source
        self.shooter_height_cm = shooter_height_cm
        self.output_path = output_path
        self.cap = None
        self.out = None
        
        # Analysis state
        self.stage = None
        self.counter = 0
        self.is_running = False
        
        # Jump Height Calculation
        self.pixels_per_cm = None
        self.standing_ankle_y = None
        self.max_jump_y = None # In pixel coordinates (smaller is higher)
        self.current_jump_height_cm = 0.0
        
        # Environment Annotations
        self.hoop_roi = None # (x, y, w, h)
        self.floor_y = None # y-coordinate
        
        # Data Recording
        self.shot_data = [] # List of dicts
        self.current_shot_stats = {}

    def start_capture(self):
        # Robust camera/video initialization
        if isinstance(self.video_source, str):
            print(f"Attempting to open video file: {self.video_source}")
            self.cap = cv2.VideoCapture(self.video_source)
        else:
            print(f"Attempting to open camera {self.video_source} with DirectShow...")
            self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                print(f"DirectShow failed. Attempting default backend for camera {self.video_source}...")
                self.cap = cv2.VideoCapture(self.video_source)

            if not self.cap.isOpened() and self.video_source == 0:
                 print(f"Failed to open camera 0. Trying camera index 1...")
                 self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
             print("CRITICAL ERROR: Could not open video source.")
             return False
        
        return True

    def setup_environment(self):
        """
        Allows user to manually select the Hoop and Floor level.
        """
        print("Setting up environment...")
        ret, frame = self.cap.read()
        if not ret:
            print("Could not read frame for setup.")
            return

        # 1. Select Hoop
        print("Select the Basketball Hoop (Draw a box and press ENTER)")
        self.hoop_roi = cv2.selectROI("Select Hoop (Press ENTER)", frame, showCrosshair=True)
        cv2.destroyWindow("Select Hoop (Press ENTER)")
        
        # 2. Select Floor
        print("Click on the Floor level (Press any key to confirm)")
        
        def click_floor(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.floor_y = y
                print(f"Floor level selected at Y={y}")
                # Draw line to visualize
                temp_frame = frame.copy()
                cv2.line(temp_frame, (0, y), (frame.shape[1], y), (0, 255, 255), 2)
                cv2.imshow("Select Floor Level (Click)", temp_frame)

        cv2.imshow("Select Floor Level (Click)", frame)
        cv2.setMouseCallback("Select Floor Level (Click)", click_floor)
        cv2.waitKey(0)
        cv2.destroyWindow("Select Floor Level (Click)")
        
        # Reset video to start if it's a file
        if isinstance(self.video_source, str):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def detect_ball(self, image):
        """
        Simple color-based ball detection (Orange).
        Returns (x, y, radius) or None.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for orange color
        # These values might need tuning based on lighting
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([25, 255, 255])
        
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Morphological operations to remove noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        center = None
        if len(cnts) > 0:
            # Find the largest contour
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            # Only consider it a ball if it's big enough but not too big
            if radius > 5 and radius < 100:
                return (int(x), int(y), int(radius))
        
        return None

    def calculate_angle(self, a, b, c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def calibrate_height(self, landmarks, image_height):
        # Estimate pixel height of the person (Ankle to Nose/Eye)
        # Using Nose as top point approximation
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        ankle_r = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        ankle_l = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        # Average ankle Y
        ankle_y = (ankle_r.y + ankle_l.y) / 2
        
        # Height in normalized coords (0-1)
        person_pixel_height_norm = abs(ankle_y - nose.y)
        
        # Convert to pixels
        person_pixel_height = person_pixel_height_norm * image_height
        
        if person_pixel_height > 0:
            self.pixels_per_cm = person_pixel_height / self.shooter_height_cm
            self.standing_ankle_y = ankle_y * image_height
            print(f"Calibration Complete: {self.pixels_per_cm:.2f} pixels/cm")
            return True
        return False

    def run(self):
        if not self.start_capture():
            return

        # Setup Environment (Hoop, Floor)
        self.setup_environment()

        # Setup Video Writer if output path is provided
        if self.output_path:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30.0
            
            # Try mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            print(f"Recording output to: {self.output_path}")

        self.is_running = True
        print("Starting analysis... Press 'q' to quit.")
        
        frame_count = 0
        calibrated = False
        
        while self.cap.isOpened() and self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video.")
                break
            
            frame_count += 1
            image_height, image_width, _ = frame.shape

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = self.pose.process(image)
            
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # --- Draw Environment Annotations ---
            # 1. Hoop
            if self.hoop_roi and self.hoop_roi[2] > 0: # w > 0
                x, y, w, h = [int(v) for v in self.hoop_roi]
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(image, "HOOP", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 2. Floor
            if self.floor_y:
                cv2.line(image, (0, self.floor_y), (image_width, self.floor_y), (0, 255, 255), 2)
                cv2.putText(image, "FLOOR", (10, self.floor_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 3. Ball Detection
            ball = self.detect_ball(image)
            if ball:
                bx, by, br = ball
                cv2.circle(image, (bx, by), br, (0, 165, 255), 2)
                cv2.circle(image, (bx, by), 2, (0, 0, 255), 3)
                cv2.putText(image, "BALL", (bx - 10, by - br - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # --- Calibration (First 30 frames or until stable) ---
                if not calibrated and frame_count > 10:
                    calibrated = self.calibrate_height(landmarks, image_height)

                # --- Coordinates ---
                shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                # --- Angles ---
                elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                knee_angle = self.calculate_angle(hip, knee, ankle)
                
                # --- Jump Height Tracking ---
                if calibrated:
                    current_ankle_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * image_height
                    # Note: Y increases downwards. Lower Y value means higher position.
                    displacement = self.standing_ankle_y - current_ankle_y
                    
                    if displacement > 0:
                        self.current_jump_height_cm = displacement / self.pixels_per_cm
                    else:
                        self.current_jump_height_cm = 0
                
                # --- Shooting Logic ---
                # "Set" phase: Elbow bent (< 100)
                # "Release" phase: Elbow extends (> 160)
                
                if elbow_angle < 90:
                    if self.stage != "set":
                        self.stage = "set"
                        self.current_shot_stats = {
                            "set_elbow_angle": elbow_angle,
                            "set_knee_angle": knee_angle,
                            "max_jump_height": 0
                        }
                    else:
                        # Update min angles during set
                        self.current_shot_stats["set_elbow_angle"] = min(self.current_shot_stats.get("set_elbow_angle", 180), elbow_angle)
                        self.current_shot_stats["set_knee_angle"] = min(self.current_shot_stats.get("set_knee_angle", 180), knee_angle)

                elif elbow_angle > 160 and self.stage == "set":
                    self.stage = "release"
                    self.counter += 1
                    
                    # Record Shot Data
                    shot_record = {
                        "shot_id": self.counter,
                        "timestamp": time.strftime("%H:%M:%S"),
                        "shooter_height_cm": self.shooter_height_cm,
                        "set_elbow_angle": int(self.current_shot_stats.get("set_elbow_angle", 0)),
                        "set_knee_angle": int(self.current_shot_stats.get("set_knee_angle", 0)),
                        "release_elbow_angle": int(elbow_angle),
                        "jump_height_cm": round(self.current_shot_stats.get("max_jump_height", 0), 2)
                    }
                    self.shot_data.append(shot_record)
                    print(f"Shot #{self.counter} Recorded: Jump {shot_record['jump_height_cm']}cm")

                # Track max jump during the shot motion
                if self.stage in ["set", "release"]:
                    if self.current_jump_height_cm > self.current_shot_stats.get("max_jump_height", 0):
                        self.current_shot_stats["max_jump_height"] = self.current_jump_height_cm

                # --- Visualization ---
                # Angles
                cv2.putText(image, f"Elbow: {int(elbow_angle)}", 
                           tuple(np.multiply(elbow, [image_width, image_height]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Stats Box
                cv2.rectangle(image, (0,0), (250,150), (245,117,16), -1)
                
                # Shot Count
                cv2.putText(image, f'SHOTS: {self.counter}', (10,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage
                cv2.putText(image, f'STAGE: {self.stage}', (10,70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                
                # Jump Height
                cv2.putText(image, f'JUMP: {int(self.current_jump_height_cm)} cm', (10,110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

                # Feedback
                if self.stage == "set" and elbow_angle > 100:
                     cv2.putText(image, "Bend Elbow More!", (250, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except Exception as e:
                pass
            
            # Render detections
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            cv2.imshow('Basketball Shooting Analysis', image)
            
            # Write frame to output video
            if self.out:
                self.out.write(image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        if self.out:
            self.out.release()
            print("Output video saved.")
            
        cv2.destroyAllWindows()
        self.save_to_csv()

    def save_to_csv(self):
        if not self.shot_data:
            print("No shots recorded.")
            return
            
        filename = "shooting_data.csv"
        keys = self.shot_data[0].keys()
        
        # Check if file exists to append or write header
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            if not file_exists:
                dict_writer.writeheader()
            dict_writer.writerows(self.shot_data)
            
        print(f"Data saved to {filename}")
        messagebox.showinfo("Success", f"Data saved to {filename}")

class ShootingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Basketball Shooting Analyzer")
        self.root.geometry("400x350")
        
        self.video_path = 0 # Default to webcam
        self.shooter_height = tk.DoubleVar(value=180.0)
        self.save_video_var = tk.BooleanVar(value=False)
        
        # UI Elements
        tk.Label(root, text="Basketball Shooting Analyzer", font=("Arial", 16)).pack(pady=10)
        
        # Height Input
        frame_height = tk.Frame(root)
        frame_height.pack(pady=10)
        tk.Label(frame_height, text="Shooter Height (cm):").pack(side=tk.LEFT)
        tk.Entry(frame_height, textvariable=self.shooter_height, width=10).pack(side=tk.LEFT, padx=5)
        
        # Video Selection
        self.lbl_video = tk.Label(root, text="Source: Webcam (Default)")
        self.lbl_video.pack(pady=5)
        
        tk.Button(root, text="Select Video File", command=self.select_video).pack(pady=5)
        tk.Button(root, text="Use Webcam", command=self.use_webcam).pack(pady=5)
        
        # Save Video Checkbox
        tk.Checkbutton(root, text="Save Analyzed Video", variable=self.save_video_var).pack(pady=5)
        
        # Start Button
        tk.Button(root, text="START ANALYSIS", command=self.start_analysis, 
                 bg="green", fg="white", font=("Arial", 12, "bold")).pack(pady=20)
        
        tk.Label(root, text="Press 'q' in the video window to stop.").pack(side=tk.BOTTOM, pady=10)

    def select_video(self):
        filename = filedialog.askopenfilename(title="Select Video", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if filename:
            self.video_path = filename
            self.lbl_video.config(text=f"Source: {os.path.basename(filename)}")

    def use_webcam(self):
        self.video_path = 0
        self.lbl_video.config(text="Source: Webcam (Default)")

    def start_analysis(self):
        try:
            height = self.shooter_height.get()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid height.")
            return
            
        output_path = None
        if self.save_video_var.get():
            output_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4")],
                title="Save Output Video As"
            )
            if not output_path:
                return # User cancelled
        
        self.root.withdraw() # Hide GUI
        analyzer = BasketballShotAnalyzer(
            video_source=self.video_path, 
            shooter_height_cm=height,
            output_path=output_path
        )
        analyzer.run()
        self.root.deiconify() # Show GUI again

if __name__ == "__main__":
    root = tk.Tk()
    app = ShootingApp(root)
    root.mainloop()
