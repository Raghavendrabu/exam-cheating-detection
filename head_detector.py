import cv2
import mediapipe as mp
import numpy as np

class HeadPoseDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_pose(self, frame):
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return "No Face"

        # Logic to calculate rotation (simplified for proctoring)
        # We check if the nose bridge is too far left or right
        for face_landmarks in results.multi_face_landmarks:
            nose = face_landmarks.landmark[1] # Nose tip
            if nose.x < 0.4: return "Looking Right"
            if nose.x > 0.6: return "Looking Left"
            return "Forward"