import cv2
from ultralytics import YOLO

class PhoneDetector:
    def __init__(self):
        # This will download 'yolov8n.pt' (6MB) on the first run
        self.model = YOLO('yolov8n.pt') 
        # We only care about 'cell phone', which is ID 67 in the COCO dataset
        self.phone_class_id = 67 

    def detect_phone(self, frame):
        """Returns True if a phone is found, and the bounding box."""
        results = self.model(frame, verbose=False, conf=0.5)[0]
        
        for box in results.boxes:
            if int(box.cls[0]) == self.phone_class_id:
                # Return the coordinates of the phone
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                return True, (x1, y1, x2, y2)
        
        return False, None