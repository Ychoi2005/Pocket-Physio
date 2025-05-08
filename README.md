# Pocket-Physio



import cv2
import mediapipe as mp
import os
import pandas as pd
from tqdm import tqdm

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
video_folder = 'PushupVideos'  # 여기에 .mp4 영상 넣기
output_folder = 'PoseData'
os.makedirs(output_folder, exist_ok=True)

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    all_landmarks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            frame_landmarks = []
            for lm in result.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            all_landmarks.append(frame_landmarks)
    cap.release()
    return all_landmarks

for filename in tqdm(os.listdir(video_folder)):
    if filename.endswith('.mp4'):
        video_path = os.path.join(video_folder, filename)
        landmarks = extract_keypoints(video_path)
        if landmarks:
            df = pd.DataFrame(landmarks)
            csv_name = filename.replace('.mp4', '.csv')
            df.to_csv(os.path.join(output_folder, csv_name), index=False)
