import cv2

def check_behavior(faces, gray_frame, detector_instance):
    status_msg = ""
    points = 0
    
    if len(faces) == 0:
        status_msg = "No Face Detected"
        points = 2
    elif len(faces) > 1:
        status_msg = "Multiple Faces Detected"
        points = 5
    elif len(faces) == 1:
        (x, y, w, h) = faces[0]
        roi_gray = gray_frame[y:y+h, x:x+w]
        eyes = detector_instance.get_eyes(roi_gray)
        if len(eyes) == 0:
            status_msg = "Eyes Not Detected"
            points = 1
            
    return status_msg, points