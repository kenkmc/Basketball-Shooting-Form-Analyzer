# Basketball Shooting Form Analyzer

This program uses computer vision (MediaPipe and OpenCV) to analyze a basketball player's shooting form in real-time. It includes a GUI for easy setup, data recording, and advanced environment analysis.

## Features

- **Graphical User Interface (GUI)**: Easy-to-use window to select video source and enter shooter details.
- **Real-time Pose Detection**: Detects body landmarks (shoulder, elbow, wrist, hip, knee, ankle).
- **Shot Analysis**:
  - **Angle Calculation**: Tracks elbow and knee angles during the shot.
  - **Shot Detection**: Automatically counts shots based on 'Set' and 'Release' phases.
  - **Jump Height Estimation**: Calculates vertical jump height (requires shooter height input).
- **Environment Annotation**:
  - **Hoop Detection**: Manually select the hoop location.
  - **Floor Level**: Manually set the floor level for reference.
  - **Ball Tracking**: Automatic color-based detection of the basketball.
- **Data Recording**: Saves detailed statistics for every shot to shooting_data.csv.
- **Video Export**: Option to save the analyzed video with all overlays.

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
2. **GUI Setup**:
   - Enter the **Shooter's Height** (in cm).
   - Select **Video Source** (Webcam or Video File).
   - Check **Save Analyzed Video** if you want to export the result.
   - Click **START ANALYSIS**.

3. **Environment Setup** (First Frame):
   - **Select Hoop**: Draw a box around the hoop and press ENTER.
   - **Select Floor**: Click on the floor level line and press any key to confirm.

4. **Analysis**:
   - Perform shooting motions.
   - The program will track your form, count shots, and estimate jump height.
   - Press q to stop and save data.

## Output

- **shooting_data.csv**: Contains timestamp, shot ID, elbow/knee angles, and jump height for each shot.
- **Saved Video**: (Optional) An .mp4 file with all visual annotations.
