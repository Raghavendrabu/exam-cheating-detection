import cv2

class ProctorDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def get_faces(self, gray_frame):
        # Increased minNeighbors from 5 to 10 to stop detecting noses as faces
        return self.face_cascade.detectMultiScale(
        gray_frame, 
        scaleFactor=1.1, 
        minNeighbors=7, 
        minSize=(120, 120) # Ignore tiny background objects that look like faces
    )

    def get_face_feature(self, gray_frame, face_coords):
        (x, y, w, h) = face_coords
    #   Crop the face and resize it to a standard size for comparison
        face_img = gray_frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        return face_img

    def get_eyes(self, face_roi):
        # Adjusted eye detection parameters for better accuracy
        return self.eye_cascade.detectMultiScale(face_roi, 1.1, 5)