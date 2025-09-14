import cv2
import mediapipe as mp
import numpy as np
import os
import json
import pandas as pd

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- Landmark & Angle Definitions ---
# This dictionary defines which landmarks to use for each angle.
# Format: "AngleName": [landmark1, landmark2, landmark3]
# landmark1 is the point on the limb, landmark2 is the joint (vertex), landmark3 is the other limb point.
# The angle is calculated at landmark2.
# You can add or remove angles as you see fit.
LANDMARK_DEFINITION = {
    "left_knee": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE],
    "right_knee": [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE],
    "left_hip": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE],
    "right_hip": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE],
    "left_elbow": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
    "right_elbow": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
    "left_shoulder": [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP],
    "right_shoulder": [mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP]
}

# --- Helper Function to Calculate Angles ---
def calculate_angle(a, b, c):
    """Calculates the angle between three landmarks."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- Main Processing Logic ---
def process_dataset(dataset_path):
    pose_data = []
    pose_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    print(f"Found {len(pose_folders)} poses: {', '.join(pose_folders)}")

    for pose_name in pose_folders:
        print(f"\nProcessing pose: {pose_name}...")
        pose_folder_path = os.path.join(dataset_path, pose_name)
        
        # Dictionary to store all calculated angles for this pose
        # Format: {"left_knee": [angle1, angle2, ...], "right_knee": [angle1, angle2, ...]}
        all_angles = {key: [] for key in LANDMARK_DEFINITION.keys()}
        
        image_files = [f for f in os.listdir(pose_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_name in image_files:
            image_path = os.path.join(pose_folder_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Process the image with MediaPipe
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Calculate each defined angle
                for angle_name, points in LANDMARK_DEFINITION.items():
                    p1, p2, p3 = points
                    # Check if landmarks are detected with sufficient visibility
                    if landmarks[p1.value].visibility > 0.7 and \
                       landmarks[p2.value].visibility > 0.7 and \
                       landmarks[p3.value].visibility > 0.7:
                        angle = calculate_angle(landmarks[p1.value], landmarks[p2.value], landmarks[p3.value])
                        all_angles[angle_name].append(angle)
        
        # --- Average the angles and create the landmark JSON ---
        landmark_json = []
        for angle_name, angle_list in all_angles.items():
            if angle_list: # Only add if we have data for this angle
                avg_angle = np.mean(angle_list)
                p1, p2, p3 = LANDMARK_DEFINITION[angle_name]
                landmark_json.append({
                    "a": p1.value,
                    "b": p2.value,
                    "c": p3.value,
                    "angle": round(avg_angle, 2),
                    "name": angle_name.replace("_", " ").title(),
                    "weight": 1.5 # You can assign weights here if needed
                })
        
        if landmark_json:
            pose_data.append({
                "name": pose_name.replace("2", " II").title(), # Clean up names like "warrior2"
                "image_url": "placeholder_url", # You'll need to update this manually
                "landmarks": json.dumps(landmark_json)
            })
        print(f"Finished processing {pose_name}. Found {len(image_files)} images.")

    return pd.DataFrame(pose_data)

if __name__ == "__main__":
    # IMPORTANT: Update these paths to match your folder structure
    # IMPORTANT: Update these paths to match your folder structure
    dataset_path = r'C:\Users\donay\Downloads\dataset\train'
    output_csv_path = r'C:\Users\donay\Downloads\dataset\yoga_poses_generated.csv'

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    print("Starting dataset processing...")
    df = process_dataset(dataset_path)
    
    if not df.empty:
        df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully generated pose data!")
        print(f"CSV file saved to: {output_csv_path}")
    else:
        print("\nNo data was generated. Please check your dataset path and images.")