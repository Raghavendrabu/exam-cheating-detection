import cv2
import time

def log_violation(frame, message):
    """Saves the violation details to a text file and captures a screenshot."""
    timestamp = int(time.time())
    readable_time = time.ctime()
    
    # 1. Update the text log
    with open("logs/exam_log.txt", "a") as f:
        f.write(f"[{readable_time}] ALERT: {message}\n")
    
    # 2. Save the screenshot
    # We clean the message for the filename (remove spaces)
    clean_msg = message.replace(" ", "_")
    filename = f"screenshots/{clean_msg}_{timestamp}.png"
    cv2.imwrite(filename, frame)
    
    print(f"Violation Logged: {message}")