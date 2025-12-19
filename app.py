"""
Basketball Shooting Analyzer - Web Application
Flask-based web interface for real-time basketball shooting analysis
"""

from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import json
from collections import deque
from datetime import datetime
import threading
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Global analyzer instance
analyzer = None
analysis_running = False
current_frame = None
frame_lock = threading.Lock()
video_source = 0  # 0 for webcam, or file path for video file
video_file_path = None

# Video playback control
video_cap = None
video_info = {'total_frames': 0, 'fps': 30, 'current_frame': 0, 'duration': 0}
seek_to_frame = None
is_paused = False


class WebBasketballAnalyzer:
    def __init__(self, shooter_height_cm=180):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.shooter_height_cm = shooter_height_cm
        self.cap = None
        
        # Analysis state
        self.stage = None
        self.counter = 0
        self.is_running = False
        
        # Smoothing buffers
        self.elbow_angle_buffer = deque(maxlen=5)
        self.knee_angle_buffer = deque(maxlen=5)
        
        # Jump Height Calculation
        self.pixels_per_cm = None
        self.standing_ankle_y = None
        self.current_jump_height_cm = 0.0
        self.calibration_frames = []
        
        # Environment Annotations
        self.hoop_roi = None
        self.floor_y = None
        
        # Ball tracking
        self.ball_trajectory = deque(maxlen=30)
        
        # Shot detection
        self.shot_start_time = None
        self.min_shot_duration = 0.3
        
        # Form rating
        self.form_score = 0
        
        # Data Recording
        self.shot_data = []
        self.current_shot_stats = {}
        
        # Real-time stats for web display
        self.current_stats = {
            'shots': 0,
            'stage': 'Ready',
            'elbow_angle': 0,
            'knee_angle': 0,
            'jump_height': 0,
            'form_score': 0,
            'feedback': []
        }
        
        # Colors (BGR)
        self.COLORS = {
            'primary': (255, 147, 0),
            'success': (0, 200, 83),
            'warning': (0, 191, 255),
            'danger': (0, 0, 255),
            'info': (255, 191, 0),
            'white': (255, 255, 255),
            'dark': (40, 40, 40)
        }

    def set_hoop_roi(self, x, y, w, h):
        """Set hoop region of interest from web interface."""
        self.hoop_roi = (x, y, w, h)
        print(f"Hoop ROI set: {self.hoop_roi}")

    def set_floor_level(self, y):
        """Set floor level from web interface."""
        self.floor_y = y
        print(f"Floor level set: {self.floor_y}")

    def detect_ball(self, image):
        """Enhanced color-based ball detection."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_orange1 = np.array([5, 100, 100])
        upper_orange1 = np.array([25, 255, 255])
        lower_orange2 = np.array([0, 80, 80])
        upper_orange2 = np.array([15, 255, 200])
        lower_orange3 = np.array([15, 100, 150])
        upper_orange3 = np.array([30, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)
        mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)
        mask3 = cv2.inRange(hsv, lower_orange3, upper_orange3)
        mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        if len(cnts) > 0:
            valid_contours = []
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 200 or area > 50000:
                    continue
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.5:
                    valid_contours.append((c, area, circularity))
            
            if valid_contours:
                valid_contours.sort(key=lambda x: x[1] * x[2], reverse=True)
                best_contour = valid_contours[0][0]
                ((x, y), radius) = cv2.minEnclosingCircle(best_contour)
                
                if 10 < radius < 80:
                    self.ball_trajectory.append((int(x), int(y)))
                    return (int(x), int(y), int(radius))
        
        return None

    def draw_ball_trajectory(self, image):
        """Draw ball trajectory."""
        if len(self.ball_trajectory) < 2:
            return
        
        for i in range(1, len(self.ball_trajectory)):
            thickness = int(np.sqrt(64 / float(len(self.ball_trajectory) - i + 1)) * 2)
            alpha = i / len(self.ball_trajectory)
            color = (0, int(165 * alpha), int(255 * alpha))
            cv2.line(image, self.ball_trajectory[i-1], self.ball_trajectory[i], color, thickness)

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points."""
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        c = np.array(c, dtype=np.float64)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle

    def get_smoothed_angle(self, angle, buffer):
        """Apply smoothing to angles."""
        buffer.append(angle)
        if len(buffer) < 2:
            return angle
        
        weights = np.exp(np.linspace(0, 1, len(buffer)))
        weights /= weights.sum()
        return np.average(list(buffer), weights=weights)

    def calculate_form_score(self, elbow_angle, knee_angle, wrist_above_shoulder):
        """Calculate shooting form score."""
        score = 0
        feedback = []
        
        if 70 <= elbow_angle <= 90:
            score += 35
        elif 60 <= elbow_angle <= 100:
            score += 25
            feedback.append("Elbow angle slightly off")
        else:
            score += 10
            feedback.append("Adjust elbow angle")
        
        if 120 <= knee_angle <= 150:
            score += 35
        elif 110 <= knee_angle <= 160:
            score += 25
            feedback.append("Adjust knee bend")
        else:
            score += 10
            feedback.append("Check leg position")
        
        if wrist_above_shoulder:
            score += 30
        else:
            score += 10
            feedback.append("Release higher")
        
        return score, feedback

    def calibrate_height(self, landmarks, image_height):
        """Calibrate height measurement."""
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        ankle_r = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        ankle_l = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        if nose.visibility < 0.7 or ankle_r.visibility < 0.7 or ankle_l.visibility < 0.7:
            return False
        
        ankle_y = (ankle_r.y + ankle_l.y) / 2
        person_pixel_height_norm = abs(ankle_y - nose.y)
        person_pixel_height = person_pixel_height_norm * image_height
        
        if person_pixel_height > 50:
            self.calibration_frames.append({
                'pixels_per_cm': person_pixel_height / self.shooter_height_cm,
                'standing_ankle_y': ankle_y * image_height
            })
            
            if len(self.calibration_frames) >= 10:
                avg_ppc = np.median([f['pixels_per_cm'] for f in self.calibration_frames])
                avg_ankle = np.median([f['standing_ankle_y'] for f in self.calibration_frames])
                
                self.pixels_per_cm = avg_ppc
                self.standing_ankle_y = avg_ankle
                return True
        
        return False

    def process_frame(self, frame):
        """Process a single frame and return annotated image."""
        image_height, image_width, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw environment annotations
        if self.hoop_roi and self.hoop_roi[2] > 0:
            x, y, w, h = [int(v) for v in self.hoop_roi]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(image, "HOOP", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if self.floor_y:
            cv2.line(image, (0, self.floor_y), (image_width, self.floor_y), (0, 255, 255), 2)
            cv2.putText(image, "FLOOR", (10, self.floor_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Ball detection
        ball = self.detect_ball(image)
        self.draw_ball_trajectory(image)
        if ball:
            bx, by, br = ball
            cv2.circle(image, (bx, by), br + 4, (0, 100, 200), 2)
            cv2.circle(image, (bx, by), br, self.COLORS['primary'], 3)
            cv2.circle(image, (bx, by), 3, self.COLORS['danger'], -1)
        
        # Process pose
        feedback_msgs = []
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Calibration
            if self.pixels_per_cm is None:
                self.calibrate_height(landmarks, image_height)
            
            # Get coordinates
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
            
            # Calculate angles
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            knee_angle = self.calculate_angle(hip, knee, ankle)
            elbow_angle = self.get_smoothed_angle(elbow_angle, self.elbow_angle_buffer)
            knee_angle = self.get_smoothed_angle(knee_angle, self.knee_angle_buffer)
            
            # Jump height
            if self.pixels_per_cm:
                current_ankle_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * image_height
                displacement = self.standing_ankle_y - current_ankle_y
                self.current_jump_height_cm = max(0, displacement / self.pixels_per_cm)
            
            # Check wrist position
            wrist_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            shoulder_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            wrist_above_shoulder = wrist_y < shoulder_y
            
            # Shot detection logic
            current_time = time.time()
            
            if elbow_angle < 100 and knee_angle < 170:
                if self.stage != "set":
                    self.stage = "set"
                    self.shot_start_time = current_time
                    self.current_shot_stats = {
                        "set_elbow_angle": elbow_angle,
                        "set_knee_angle": knee_angle,
                        "max_jump_height": 0
                    }
                else:
                    self.current_shot_stats["set_elbow_angle"] = min(
                        self.current_shot_stats.get("set_elbow_angle", 180), elbow_angle)
                    self.current_shot_stats["set_knee_angle"] = min(
                        self.current_shot_stats.get("set_knee_angle", 180), knee_angle)
            
            elif elbow_angle > 150 and self.stage == "set":
                shot_duration = current_time - self.shot_start_time if self.shot_start_time else 0
                
                if shot_duration >= self.min_shot_duration:
                    self.stage = "release"
                    self.counter += 1
                    
                    form_score, feedback = self.calculate_form_score(
                        self.current_shot_stats.get("set_elbow_angle", 90),
                        self.current_shot_stats.get("set_knee_angle", 140),
                        wrist_above_shoulder
                    )
                    self.form_score = form_score
                    
                    shot_record = {
                        "shot_id": self.counter,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "set_elbow_angle": int(self.current_shot_stats.get("set_elbow_angle", 0)),
                        "set_knee_angle": int(self.current_shot_stats.get("set_knee_angle", 0)),
                        "release_elbow_angle": int(elbow_angle),
                        "jump_height_cm": round(self.current_shot_stats.get("max_jump_height", 0), 2),
                        "form_score": form_score
                    }
                    self.shot_data.append(shot_record)
                else:
                    self.stage = None
            
            # Track max jump
            if self.stage in ["set", "release"]:
                if self.current_jump_height_cm > self.current_shot_stats.get("max_jump_height", 0):
                    self.current_shot_stats["max_jump_height"] = self.current_jump_height_cm
            
            # Update current stats for web display
            self.current_stats = {
                'shots': self.counter,
                'stage': self.stage or 'Ready',
                'elbow_angle': int(elbow_angle),
                'knee_angle': int(knee_angle),
                'jump_height': int(self.current_jump_height_cm),
                'form_score': self.form_score,
                'feedback': feedback_msgs
            }
            
            # Draw angle labels
            elbow_pos = tuple(np.multiply(elbow, [image_width, image_height]).astype(int))
            knee_pos = tuple(np.multiply(knee, [image_width, image_height]).astype(int))
            cv2.putText(image, f"{int(elbow_angle)}", (elbow_pos[0] + 10, elbow_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['info'], 2)
            cv2.putText(image, f"{int(knee_angle)}", (knee_pos[0] + 10, knee_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['info'], 2)
            
            # Feedback
            if self.stage == "set":
                if elbow_angle > 100:
                    feedback_msgs.append("Bend elbow more")
                if knee_angle > 165:
                    feedback_msgs.append("Bend knees")
                if elbow_angle <= 90 and knee_angle <= 150:
                    feedback_msgs.append("Good form!")
            
            self.current_stats['feedback'] = feedback_msgs
            
        except Exception as e:
            pass
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2, circle_radius=3),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 200, 83), thickness=2, circle_radius=2)
            )
        
        return image

    def get_shot_data(self):
        """Return shot data as JSON."""
        return self.shot_data

    def reset(self):
        """Reset analyzer state."""
        self.stage = None
        self.counter = 0
        self.shot_data = []
        self.form_score = 0
        self.calibration_frames = []
        self.pixels_per_cm = None
        self.ball_trajectory.clear()
        self.elbow_angle_buffer.clear()
        self.knee_angle_buffer.clear()


def generate_frames():
    """Generate video frames for streaming."""
    global analyzer, analysis_running, current_frame, video_source, video_file_path
    global video_cap, video_info, seek_to_frame, is_paused
    
    # Open video source (webcam or file)
    if video_file_path and os.path.exists(video_file_path):
        print(f"Opening video file: {video_file_path}")
        cap = cv2.VideoCapture(video_file_path)
    else:
        print("Opening webcam...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open video source")
        return
    
    video_cap = cap
    
    # Set resolution for webcam
    if not video_file_path:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Get video properties for proper playback speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_delay = 1.0 / fps
    
    # Store video info for seeking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_info['total_frames'] = total_frames
    video_info['fps'] = fps
    video_info['duration'] = total_frames / fps if fps > 0 else 0
    video_info['current_frame'] = 0
    
    last_frame = None
    
    while analysis_running:
        start_time = time.time()
        
        # Handle seeking
        if seek_to_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to_frame)
            video_info['current_frame'] = seek_to_frame
            seek_to_frame = None
        
        # If paused, yield the last frame
        if is_paused and last_frame is not None:
            ret, buffer = cv2.imencode('.jpg', last_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)  # Small delay when paused
            continue
        
        ret, frame = cap.read()
        
        if not ret:
            # If video file ended, loop back to start
            if video_file_path:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                video_info['current_frame'] = 0
                continue
            break
        
        # Update current frame position
        video_info['current_frame'] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if analyzer:
            frame = analyzer.process_frame(frame)
        
        last_frame = frame.copy()
        
        with frame_lock:
            current_frame = frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Control playback speed for video files
        if video_file_path:
            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
    
    video_cap = None
    cap.release()


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    """Start the analysis."""
    global analyzer, analysis_running, video_file_path
    
    data = request.get_json()
    height = data.get('height', 180)
    source = data.get('source', 'webcam')
    
    # Set video source
    if source == 'webcam':
        video_file_path = None
    # If source is a filename, it was uploaded
    
    analyzer = WebBasketballAnalyzer(shooter_height_cm=float(height) if height else 180)
    analysis_running = True
    
    return jsonify({'status': 'started'})


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Upload a video file for analysis."""
    global video_file_path
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    video_file_path = filepath
    print(f"Video uploaded: {filepath}")
    
    return jsonify({'status': 'uploaded', 'filename': filename})


@app.route('/use_webcam', methods=['POST'])
def use_webcam():
    """Switch to webcam source."""
    global video_file_path
    video_file_path = None
    return jsonify({'status': 'ok'})


@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    """Stop the analysis."""
    global analysis_running
    analysis_running = False
    return jsonify({'status': 'stopped'})


@app.route('/set_hoop', methods=['POST'])
def set_hoop():
    """Set hoop ROI."""
    global analyzer
    data = request.get_json()
    if analyzer:
        analyzer.set_hoop_roi(
            int(data['x']), int(data['y']),
            int(data['width']), int(data['height'])
        )
    return jsonify({'status': 'ok'})


@app.route('/set_floor', methods=['POST'])
def set_floor():
    """Set floor level."""
    global analyzer
    data = request.get_json()
    if analyzer:
        analyzer.set_floor_level(int(data['y']))
    return jsonify({'status': 'ok'})


@app.route('/get_stats')
def get_stats():
    """Get current statistics."""
    global analyzer
    if analyzer:
        return jsonify(analyzer.current_stats)
    return jsonify({
        'shots': 0,
        'stage': 'Not Started',
        'elbow_angle': 0,
        'knee_angle': 0,
        'jump_height': 0,
        'form_score': 0,
        'feedback': []
    })


@app.route('/get_shot_data')
def get_shot_data():
    """Get all shot data."""
    global analyzer
    if analyzer:
        return jsonify(analyzer.get_shot_data())
    return jsonify([])


@app.route('/download_csv')
def download_csv():
    """Download shot data as CSV."""
    global analyzer
    if not analyzer or not analyzer.shot_data:
        return jsonify({'error': 'No data available'})
    
    filename = f"shooting_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    keys = analyzer.shot_data[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(analyzer.shot_data)
    
    return send_file(filepath, as_attachment=True, download_name=filename)


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the analyzer."""
    global analyzer
    if analyzer:
        analyzer.reset()
    return jsonify({'status': 'reset'})


@app.route('/get_video_info')
def get_video_info():
    """Get video playback information."""
    global video_info, video_file_path
    return jsonify({
        'total_frames': video_info['total_frames'],
        'fps': video_info['fps'],
        'current_frame': video_info['current_frame'],
        'duration': video_info['duration'],
        'is_video_file': video_file_path is not None
    })


@app.route('/seek_video', methods=['POST'])
def seek_video():
    """Seek to a specific frame in the video."""
    global seek_to_frame, video_info
    data = request.get_json()
    
    if 'frame' in data:
        target_frame = int(data['frame'])
        target_frame = max(0, min(target_frame, video_info['total_frames'] - 1))
        seek_to_frame = target_frame
        return jsonify({'status': 'ok', 'frame': target_frame})
    elif 'position' in data:
        # Seek by percentage (0-100)
        position = float(data['position'])
        target_frame = int((position / 100.0) * video_info['total_frames'])
        target_frame = max(0, min(target_frame, video_info['total_frames'] - 1))
        seek_to_frame = target_frame
        return jsonify({'status': 'ok', 'frame': target_frame})
    
    return jsonify({'error': 'No frame or position provided'}), 400


@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    """Toggle video playback pause state."""
    global is_paused
    is_paused = not is_paused
    return jsonify({'status': 'ok', 'paused': is_paused})


@app.route('/set_pause', methods=['POST'])
def set_pause():
    """Set video playback pause state."""
    global is_paused
    data = request.get_json()
    is_paused = data.get('paused', False)
    return jsonify({'status': 'ok', 'paused': is_paused})


if __name__ == '__main__':
    print("=" * 60)
    print("Basketball Shooting Analyzer - Web Application")
    print("=" * 60)
    print("\nOpen your browser and go to: http://localhost:5000")
    print("\nWEBCAM PLACEMENT INSTRUCTIONS:")
    print("-" * 40)
    print("Place the webcam to the SIDE of the player (perpendicular")
    print("to the shooting direction), so the camera can see:")
    print("  1. The player's full body (head to feet)")
    print("  2. The player's shooting arm clearly")
    print("  3. The basketball hoop (rim and net)")
    print("\nIdeal setup:")
    print("  - Camera at waist height (1-1.2m from ground)")
    print("  - 3-5 meters away from the player")
    print("  - Player shooting towards or away from camera view")
    print("  - Good lighting, avoid backlighting")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
