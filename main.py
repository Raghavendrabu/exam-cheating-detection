import cv2
import time
import numpy as np
from utils import initialize_workspace, clear_old_data
from face_detector import ProctorDetector
from behavior_detector import check_behavior
from logger import log_violation

def main():
    initialize_workspace()
    clear_old_data() 
    detector = ProctorDetector()
    cap = cv2.VideoCapture(0)
    
    # --- LOGIC VARIABLES ---
    cheat_score = 0
    absence_start_time = None
    total_absence_seconds = 0
    last_log_time = 0
    missing_face_frames = 0
    
    # --- PHASE 1: CALIBRATION ---
    print("Align your face and press 'C' to calibrate...")
    reference_face = None

    while reference_face is None:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.get_faces(gray)
        
        for (x, y, w, h) in faces:
            # FIXED: Added (x+w, y+h) to complete the rectangle coordinates
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Press 'C' to Lock Identity", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('c') and len(faces) == 1:
            reference_face = detector.get_face_feature(gray, faces[0])
            cv2.destroyWindow("Calibration")
            print("Identity Locked!")

    # --- PHASE 2: MONITORING ---
    print("System Active. Monitoring started...")
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray) 
        faces = detector.get_faces(gray)
        
        status, points = check_behavior(faces, gray, detector)
        
        # Identity Check Logic
        if len(faces) == 1:
            current_face = detector.get_face_feature(gray, faces[0])
            # Compare current face with reference using Template Matching
            res = cv2.matchTemplate(current_face, reference_face, cv2.TM_CCOEFF_NORMED)
            similarity = res[0][0]
            
            # If the match is less than 70%, it's likely a different person
            if similarity < 0.60: 
                status = "Wrong Person Detected"
                points = 5

        # Stability Buffer
        if status == "No Face Detected":
            missing_face_frames += 1
        else:
            missing_face_frames = 0
            if status != "Wrong Person Detected":
                absence_start_time = None

        # Scoring Logic
        if status != "":
            if status == "No Face Detected":
                if missing_face_frames > 10:
                    cheat_score += 1.5
                    if absence_start_time is None: absence_start_time = time.time()
                    total_absence_seconds = int(time.time() - absence_start_time)
            else:
                cheat_score += points
        else:
            if cheat_score > 0: cheat_score -= 0.3 

        cheat_score = max(0, min(100, cheat_score))

        # Dashboard UI
        color = (0, 255, 0) # Green
        display_status = status if (missing_face_frames > 10 or (status != "" and status != "No Face Detected")) else "Normal"

        if cheat_score > 50 or total_absence_seconds > 10:
            color = (0, 0, 255) # Red
            cv2.putText(frame, "CRITICAL WARNING", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            if time.time() - last_log_time > 3:
                log_violation(frame, f"Alert_{display_status.replace(' ', '_')}")
                last_log_time = time.time()

        # Overlay Dashboard
        cv2.rectangle(frame, (0, 0), (320, 130), (0,0,0), -1) 
        cv2.putText(frame, f"Cheat Score: {int(cheat_score)}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Absence: {total_absence_seconds}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {display_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("AI Proctor Dashboard", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()