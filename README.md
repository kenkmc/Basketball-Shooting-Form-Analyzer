# Basketball Shooting Form Analyzer

This program uses computer vision (MediaPipe and OpenCV) to analyze a basketball player's shooting form in real-time.

## Features

- **Real-time Pose Detection**: Detects body landmarks (shoulder, elbow, wrist, hip, knee, ankle).
- **Angle Calculation**: Calculates and displays the elbow and knee angles.
- **Shot Detection**: Counts shots based on the sequence of "Set" (elbow bent) and "Release" (arm extended).
- **Form Feedback**: Provides visual feedback if the elbow is not bent enough during the set point.

## Requirements

- Python 3.x
- Webcam (or a video file)

## Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the script:
   ```bash
   python shooting_analysis.py
   ```
2. The program will open your webcam.
3. Stand in front of the camera so your full body (or at least upper body) is visible.
4. Perform a shooting motion.
   - **Set Point**: Bend your elbow to < 90 degrees.
   - **Release**: Extend your arm fully (> 160 degrees).
5. The counter will increment when a shot is detected.
6. Press 'q' to quit.

## Customization

- To use a video file instead of a webcam, modify the `__main__` block in `shooting_analysis.py`:
  ```python
  analyzer = BasketballShotAnalyzer("path/to/your/video.mp4")
  ```
