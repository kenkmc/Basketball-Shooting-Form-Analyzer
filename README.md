# Basketball Shooting Form Analyzer (Web App)

This application uses computer vision (MediaPipe and OpenCV) to analyze a basketball player's shooting form in real-time or from uploaded videos. It features a modern web interface for easy interaction, data visualization, and video playback control.

## Features

- **Web Interface**: Clean, responsive dashboard accessible via browser.
- **Dual Modes**:
  - **Webcam Mode**: Real-time analysis using your computer's camera.
  - **Video Upload Mode**: Upload and analyze pre-recorded video files.
- **Video Playback Control**:
  - **Timeline Slider**: Seek to any part of the video.
  - **Frame Stepping**: Move frame-by-frame for precise analysis.
  - **Slow Motion/Pause**: Control playback flow.
- **Advanced Analysis**:
  - **Pose Detection**: Tracks key body landmarks (shoulder, elbow, wrist, hip, knee, ankle).
  - **Form Scoring**: Calculates a form score based on shooting mechanics.
  - **Shot Detection**: Automatically counts shots and identifies 'Set' and 'Release' phases.
  - **Jump Height**: Estimates vertical jump height (optional shooter height input).
- **Environment Tools**:
  - **Hoop Selection**: Interactive tool to mark the hoop location.
  - **Floor Level**: Interactive tool to set the floor level.
- **Data Export**: Download detailed shot statistics as CSV.

## Requirements

- Python 3.8+
- Webcam (for live analysis)

## Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ``` 

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ``` 
2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. **Setup**:
   - **Video Source**: Choose "Use Webcam" or "Upload Video File".
   - **Shooter Height** (Optional): Enter height in cm for accurate jump height measurement.

4. **Analysis**:
   - Click **Start Analysis**.
   - **Set Hoop/Floor**: Use the buttons to mark the hoop and floor on the video feed for better context.
   - Perform shooting motions. The app will track your form and update stats in real-time.

5. **Review**:
   - Use the **Video Timeline** to review specific shots (Video File mode).
   - Check the **Live Statistics** and **Shot History** panels.
   - Click **Download CSV** to save your session data.

## Camera Placement

For best results:
- Place camera to the **SIDE** of the player (perpendicular to shooting direction).
- Camera should be at waist height (1-1.2m).
- Ensure the camera sees the player's full body and the hoop.
