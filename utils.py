import os
import cv2

def initialize_workspace():
    for folder in ["logs", "screenshots", "reference"]:
        if not os.path.exists(folder):
            os.makedirs(folder)

def clear_old_data():
    """Deletes logs and screenshots from previous sessions."""
    # Clear the text file
    if os.path.exists("logs/exam_log.txt"):
        open("logs/exam_log.txt", "w").close()
    
    # Delete images
    if os.path.exists("screenshots"):
        for file in os.listdir("screenshots"):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                os.remove(os.path.join("screenshots", file))

def load_reference_face():
    # Place a photo of yourself in the 'reference' folder named 'me.jpg'
    ref_path = "reference/me.jpg"
    if os.path.exists(ref_path):
        img = cv2.imread(ref_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
    return None