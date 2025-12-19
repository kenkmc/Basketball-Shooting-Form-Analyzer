import cv2
import mediapipe as mp
import numpy as np
import math
import csv
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from collections import deque

class BasketballShotAnalyzer:
    def __init__(self, video_source=0, shooter_height_cm=180, output_path=None):
        self.mp_pose = mp.solutions.pose
        # Higher model complexity for better accuracy
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Maximum complexity for best accuracy
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.7,  # Higher confidence threshold
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.video_source = video_source
        self.shooter_height_cm = shooter_height_cm
        self.output_path = output_path
        self.cap = None
        self.out = None
        
        # Analysis state
        self.stage = None
        self.counter = 0
        self.is_running = False
        
        # Smoothing buffers for angle calculations (reduces jitter)
        self.elbow_angle_buffer = deque(maxlen=5)
        self.knee_angle_buffer = deque(maxlen=5)
        self.wrist_height_buffer = deque(maxlen=5)
        
        # Jump Height Calculation
        self.pixels_per_cm = None
        self.standing_ankle_y = None
        self.max_jump_y = None
        self.current_jump_height_cm = 0.0
        self.calibration_frames = []
        
        # Environment Annotations
        self.hoop_roi = None  # (x, y, w, h)
        self.hoop_center = None  # (x, y) center of hoop
        self.floor_y = None
        
        # Ball tracking
        self.ball_trajectory = deque(maxlen=30)  # Store ball positions for trajectory
        self.ball_hsv_range = None  # Will be calibrated
        
        # Shot detection refinement
        self.shot_start_time = None
        self.min_shot_duration = 0.3  # Minimum 300ms for a valid shot
        self.wrist_above_shoulder_frames = 0
        
        # Form rating
        self.form_score = 0
        
        # Data Recording
        self.shot_data = []
        self.current_shot_stats = {}
        
        # UI Colors (BGR)
        self.COLORS = {
            'primary': (255, 147, 0),    # Orange
            'success': (0, 200, 83),     # Green
            'warning': (0, 191, 255),    # Yellow/Gold
            'danger': (0, 0, 255),       # Red
            'info': (255, 191, 0),       # Cyan
            'white': (255, 255, 255),
            'dark': (40, 40, 40),
            'light_bg': (245, 245, 245)
        }

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
        Enhanced color-based ball detection with multiple orange ranges.
        Returns (x, y, radius) or None.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Multiple HSV ranges for different lighting conditions
        # Standard orange basketball
        lower_orange1 = np.array([5, 100, 100])
        upper_orange1 = np.array([25, 255, 255])
        
        # Darker orange / brown tones
        lower_orange2 = np.array([0, 80, 80])
        upper_orange2 = np.array([15, 255, 200])
        
        # Lighter / yellowish orange
        lower_orange3 = np.array([15, 100, 150])
        upper_orange3 = np.array([30, 255, 255])
        
        # Combine masks
        mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)
        mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)
        mask3 = cv2.inRange(hsv, lower_orange3, upper_orange3)
        mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
        
        # Better morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Find contours
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        if len(cnts) > 0:
            # Filter contours by circularity and area
            valid_contours = []
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 200 or area > 50000:  # Filter by area
                    continue
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.5:  # Must be reasonably circular
                    valid_contours.append((c, area, circularity))
            
            if valid_contours:
                # Sort by area and circularity combined score
                valid_contours.sort(key=lambda x: x[1] * x[2], reverse=True)
                best_contour = valid_contours[0][0]
                ((x, y), radius) = cv2.minEnclosingCircle(best_contour)
                
                if 10 < radius < 80:
                    # Add to trajectory
                    self.ball_trajectory.append((int(x), int(y)))
                    return (int(x), int(y), int(radius))
        
        return None
    
    def draw_ball_trajectory(self, image):
        """Draw the ball's recent trajectory."""
        if len(self.ball_trajectory) < 2:
            return
        
        for i in range(1, len(self.ball_trajectory)):
            thickness = int(np.sqrt(64 / float(len(self.ball_trajectory) - i + 1)) * 2)
            alpha = i / len(self.ball_trajectory)
            color = (0, int(165 * alpha), int(255 * alpha))
            cv2.line(image, self.ball_trajectory[i-1], self.ball_trajectory[i], color, thickness)

    def calculate_angle(self, a, b, c):
        """Calculate angle with improved precision."""
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        c = np.array(c, dtype=np.float64)
        
        ba = a - b
        bc = c - b
        
        # Use dot product formula for more stable calculation
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    
    def get_smoothed_angle(self, angle, buffer):
        """Apply exponential moving average smoothing to angles."""
        buffer.append(angle)
        if len(buffer) < 2:
            return angle
        
        # Weighted average - more recent values have higher weight
        weights = np.exp(np.linspace(0, 1, len(buffer)))
        weights /= weights.sum()
        smoothed = np.average(list(buffer), weights=weights)
        return smoothed
    
    def calculate_form_score(self, elbow_angle, knee_angle, wrist_above_shoulder):
        """Calculate a shooting form score (0-100)."""
        score = 0
        feedback = []
        
        # Ideal elbow angle at set: 70-90 degrees
        if 70 <= elbow_angle <= 90:
            score += 35
        elif 60 <= elbow_angle <= 100:
            score += 25
            feedback.append("Elbow angle slightly off")
        else:
            score += 10
            feedback.append("Adjust elbow angle")
        
        # Ideal knee bend: 120-150 degrees (slight bend)
        if 120 <= knee_angle <= 150:
            score += 35
        elif 110 <= knee_angle <= 160:
            score += 25
            feedback.append("Adjust knee bend")
        else:
            score += 10
            feedback.append("Check leg position")
        
        # Wrist should be above shoulder during release
        if wrist_above_shoulder:
            score += 30
        else:
            score += 10
            feedback.append("Release higher")
        
        return score, feedback

    def calibrate_height(self, landmarks, image_height):
        """Multi-frame calibration for better accuracy."""
        # Get all relevant landmarks for height estimation
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        ankle_r = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        ankle_l = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        # Check visibility confidence
        if nose.visibility < 0.7 or ankle_r.visibility < 0.7 or ankle_l.visibility < 0.7:
            return False
        
        ankle_y = (ankle_r.y + ankle_l.y) / 2
        person_pixel_height_norm = abs(ankle_y - nose.y)
        person_pixel_height = person_pixel_height_norm * image_height
        
        if person_pixel_height > 50:  # Minimum reasonable height in pixels
            self.calibration_frames.append({
                'pixels_per_cm': person_pixel_height / self.shooter_height_cm,
                'standing_ankle_y': ankle_y * image_height
            })
            
            # Use average of multiple frames for stable calibration
            if len(self.calibration_frames) >= 10:
                avg_ppc = np.median([f['pixels_per_cm'] for f in self.calibration_frames])
                avg_ankle = np.median([f['standing_ankle_y'] for f in self.calibration_frames])
                
                self.pixels_per_cm = avg_ppc
                self.standing_ankle_y = avg_ankle
                print(f"Calibration Complete: {self.pixels_per_cm:.2f} pixels/cm (averaged over {len(self.calibration_frames)} frames)")
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

            # 3. Ball Detection & Trajectory
            ball = self.detect_ball(image)
            self.draw_ball_trajectory(image)
            if ball:
                bx, by, br = ball
                # Draw ball with glow effect
                cv2.circle(image, (bx, by), br + 4, (0, 100, 200), 2)
                cv2.circle(image, (bx, by), br, self.COLORS['primary'], 3)
                cv2.circle(image, (bx, by), 3, self.COLORS['danger'], -1)

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
                
                # --- Check if wrist is above shoulder ---
                wrist_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                shoulder_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                wrist_above_shoulder = wrist_y < shoulder_y
                
                # --- Smoothed Angles ---
                elbow_angle = self.get_smoothed_angle(elbow_angle, self.elbow_angle_buffer)
                knee_angle = self.get_smoothed_angle(knee_angle, self.knee_angle_buffer)
                
                # --- Improved Shooting Logic ---
                # "Set" phase: Elbow bent (< 100) AND knees slightly bent
                # "Release" phase: Elbow extends (> 150) AND wrist above shoulder
                
                current_time = time.time()
                
                if elbow_angle < 100 and knee_angle < 170:
                    if self.stage != "set":
                        self.stage = "set"
                        self.shot_start_time = current_time
                        self.current_shot_stats = {
                            "set_elbow_angle": elbow_angle,
                            "set_knee_angle": knee_angle,
                            "max_jump_height": 0,
                            "wrist_above_shoulder": False
                        }
                    else:
                        # Track minimum angles during set
                        self.current_shot_stats["set_elbow_angle"] = min(
                            self.current_shot_stats.get("set_elbow_angle", 180), elbow_angle)
                        self.current_shot_stats["set_knee_angle"] = min(
                            self.current_shot_stats.get("set_knee_angle", 180), knee_angle)

                elif elbow_angle > 150 and self.stage == "set":
                    # Validate shot duration to avoid false positives
                    shot_duration = current_time - self.shot_start_time if self.shot_start_time else 0
                    
                    if shot_duration >= self.min_shot_duration:
                        self.stage = "release"
                        self.counter += 1
                        
                        # Calculate form score
                        form_score, feedback = self.calculate_form_score(
                            self.current_shot_stats.get("set_elbow_angle", 90),
                            self.current_shot_stats.get("set_knee_angle", 140),
                            wrist_above_shoulder
                        )
                        self.form_score = form_score
                        
                        # Record Shot Data with enhanced metrics
                        shot_record = {
                            "shot_id": self.counter,
                            "timestamp": time.strftime("%H:%M:%S"),
                            "shooter_height_cm": self.shooter_height_cm,
                            "set_elbow_angle": int(self.current_shot_stats.get("set_elbow_angle", 0)),
                            "set_knee_angle": int(self.current_shot_stats.get("set_knee_angle", 0)),
                            "release_elbow_angle": int(elbow_angle),
                            "jump_height_cm": round(self.current_shot_stats.get("max_jump_height", 0), 2),
                            "form_score": form_score,
                            "shot_duration_sec": round(shot_duration, 2)
                        }
                        self.shot_data.append(shot_record)
                        print(f"Shot #{self.counter} | Form: {form_score}/100 | Jump: {shot_record['jump_height_cm']}cm")
                    else:
                        self.stage = None  # Reset if shot was too quick (false positive)

                # Track max jump during the shot motion
                if self.stage in ["set", "release"]:
                    if self.current_jump_height_cm > self.current_shot_stats.get("max_jump_height", 0):
                        self.current_shot_stats["max_jump_height"] = self.current_jump_height_cm

                # --- Enhanced Visualization ---
                # Angle labels near joints
                elbow_pos = tuple(np.multiply(elbow, [image_width, image_height]).astype(int))
                knee_pos = tuple(np.multiply(knee, [image_width, image_height]).astype(int))
                
                # Draw angle arcs
                cv2.putText(image, f"{int(elbow_angle)}°", 
                           (elbow_pos[0] + 10, elbow_pos[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['info'], 2, cv2.LINE_AA)
                cv2.putText(image, f"{int(knee_angle)}°", 
                           (knee_pos[0] + 10, knee_pos[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['info'], 2, cv2.LINE_AA)
                
                # --- Modern Stats Panel (Semi-transparent) ---
                overlay = image.copy()
                panel_height = 200
                cv2.rectangle(overlay, (0, 0), (280, panel_height), self.COLORS['dark'], -1)
                cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                
                # Header
                cv2.putText(image, "SHOT ANALYZER", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['primary'], 2, cv2.LINE_AA)
                cv2.line(image, (10, 35), (200, 35), self.COLORS['primary'], 2)
                
                # Shot Count (Large)
                cv2.putText(image, f"SHOTS", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['white'], 1, cv2.LINE_AA)
                cv2.putText(image, f"{self.counter}", (80, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.COLORS['success'], 2, cv2.LINE_AA)
                
                # Stage with color coding
                stage_color = self.COLORS['warning'] if self.stage == "set" else (
                    self.COLORS['success'] if self.stage == "release" else self.COLORS['white'])
                cv2.putText(image, f"Stage:", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['white'], 1, cv2.LINE_AA)
                cv2.putText(image, f"{self.stage or 'Ready'}", (70, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, stage_color, 2, cv2.LINE_AA)
                
                # Jump Height with bar indicator
                cv2.putText(image, f"Jump:", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['white'], 1, cv2.LINE_AA)
                cv2.putText(image, f"{int(self.current_jump_height_cm)} cm", (70, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['info'], 2, cv2.LINE_AA)
                
                # Jump height bar
                max_display_jump = 60  # cm
                jump_bar_width = int(min(self.current_jump_height_cm / max_display_jump, 1.0) * 150)
                cv2.rectangle(image, (10, 130), (160, 145), self.COLORS['dark'], -1)
                cv2.rectangle(image, (10, 130), (10 + jump_bar_width, 145), self.COLORS['info'], -1)
                
                # Form Score (if available)
                if self.form_score > 0:
                    score_color = self.COLORS['success'] if self.form_score >= 70 else (
                        self.COLORS['warning'] if self.form_score >= 50 else self.COLORS['danger'])
                    cv2.putText(image, f"Form Score:", (10, 170), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['white'], 1, cv2.LINE_AA)
                    cv2.putText(image, f"{self.form_score}/100", (110, 170), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2, cv2.LINE_AA)
                
                # Angles display
                cv2.putText(image, f"Elbow: {int(elbow_angle)}°  Knee: {int(knee_angle)}°", (10, 195), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['white'], 1, cv2.LINE_AA)

                # --- Real-time Feedback Panel (Right side) ---
                feedback_msgs = []
                if self.stage == "set":
                    if elbow_angle > 100:
                        feedback_msgs.append(("Bend elbow more", self.COLORS['warning']))
                    if knee_angle > 165:
                        feedback_msgs.append(("Bend knees", self.COLORS['warning']))
                    if elbow_angle <= 90 and knee_angle <= 150:
                        feedback_msgs.append(("Good form!", self.COLORS['success']))
                
                for i, (msg, color) in enumerate(feedback_msgs):
                    # Draw feedback with background
                    text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    x_pos = image_width - text_size[0] - 20
                    y_pos = 40 + (i * 35)
                    cv2.rectangle(image, (x_pos - 10, y_pos - 25), 
                                 (image_width - 5, y_pos + 5), color, -1)
                    cv2.putText(image, msg, (x_pos, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['white'], 2, cv2.LINE_AA)

            except Exception as e:
                pass
            
            # Render detections with custom styling
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(255, 255, 255), thickness=2, circle_radius=3),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 200, 83), thickness=2, circle_radius=2)
                )
            
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
        self.root.geometry("500x550")
        self.root.resizable(False, False)
        
        # Configure styles
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except:
            pass  # Use default theme if clam not available
        
        # Custom colors
        self.BG_COLOR = "#2C3E50"
        self.FG_COLOR = "#ECF0F1"
        self.ACCENT_COLOR = "#E67E22"
        self.SUCCESS_COLOR = "#27AE60"
        
        self.root.configure(bg=self.BG_COLOR)
        
        # Configure custom styles
        self.style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'), 
                            background=self.BG_COLOR, foreground=self.ACCENT_COLOR)
        self.style.configure('Subtitle.TLabel', font=('Segoe UI', 10), 
                            background=self.BG_COLOR, foreground=self.FG_COLOR)
        self.style.configure('TFrame', background=self.BG_COLOR)
        self.style.configure('TLabelframe', background=self.BG_COLOR)
        self.style.configure('TLabelframe.Label', background=self.BG_COLOR, foreground=self.FG_COLOR)
        self.style.configure('TLabel', background=self.BG_COLOR, foreground=self.FG_COLOR, font=('Segoe UI', 10))
        self.style.configure('TCheckbutton', background=self.BG_COLOR, foreground=self.FG_COLOR, font=('Segoe UI', 10))
        self.style.configure('TButton', font=('Segoe UI', 10))
        self.style.configure('Start.TButton', font=('Segoe UI', 14, 'bold'))
        self.style.configure('TEntry', font=('Segoe UI', 11))
        
        self.video_path = 0
        self.shooter_height = tk.DoubleVar(value=180.0)
        self.save_video_var = tk.BooleanVar(value=False)
        
        self._create_ui()
    
    def _create_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        ttk.Label(main_frame, text="Basketball Shot Analyzer", style='Title.TLabel').pack(pady=(0, 5))
        ttk.Label(main_frame, text="AI-Powered Shooting Form Analysis", style='Subtitle.TLabel').pack(pady=(0, 20))
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # --- Settings Section ---
        settings_frame = ttk.LabelFrame(main_frame, text=" Settings ", padding="10")
        settings_frame.pack(fill='x', pady=10)
        
        # Height Input
        height_frame = ttk.Frame(settings_frame)
        height_frame.pack(fill='x', pady=5)
        ttk.Label(height_frame, text="Shooter Height:").pack(side=tk.LEFT)
        height_entry = ttk.Entry(height_frame, textvariable=self.shooter_height, width=8, font=('Segoe UI', 11))
        height_entry.pack(side=tk.LEFT, padx=10)
        ttk.Label(height_frame, text="cm").pack(side=tk.LEFT)
        
        # --- Video Source Section ---
        source_frame = ttk.LabelFrame(main_frame, text=" Video Source ", padding="10")
        source_frame.pack(fill='x', pady=10)
        
        # Source display
        self.source_var = tk.StringVar(value="[Webcam] Default")
        self.lbl_video = ttk.Label(source_frame, textvariable=self.source_var, font=('Segoe UI', 10, 'italic'))
        self.lbl_video.pack(pady=5)
        
        # Buttons frame
        btn_frame = ttk.Frame(source_frame)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Select Video File", command=self.select_video, width=18).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Use Webcam", command=self.use_webcam, width=18).pack(side=tk.LEFT, padx=5)
        
        # --- Options Section ---
        options_frame = ttk.LabelFrame(main_frame, text=" Options ", padding="10")
        options_frame.pack(fill='x', pady=10)
        
        ttk.Checkbutton(options_frame, text="Save analyzed video output", 
                       variable=self.save_video_var).pack(anchor='w')
        
        # --- Start Button (Large and Prominent) ---
        start_btn = tk.Button(main_frame, text=">>> START ANALYSIS <<<", command=self.start_analysis,
                             bg=self.SUCCESS_COLOR, fg='white', font=('Segoe UI', 16, 'bold'),
                             activebackground='#229954', activeforeground='white',
                             relief='raised', cursor='hand2', pady=15, bd=3)
        start_btn.pack(fill='x', pady=25)
        
        # --- Footer ---
        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(side=tk.BOTTOM, fill='x')
        ttk.Label(footer_frame, text="Press 'Q' in the video window to stop analysis", 
                 style='Subtitle.TLabel').pack()
        ttk.Label(footer_frame, text="v2.0 | MediaPipe + OpenCV", 
                 style='Subtitle.TLabel').pack()

    def select_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video", 
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.video_path = filename
            self.source_var.set(f"[Video] {os.path.basename(filename)}")

    def use_webcam(self):
        self.video_path = 0
        self.source_var.set("[Webcam] Default")

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
