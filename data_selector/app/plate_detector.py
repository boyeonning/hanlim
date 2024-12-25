import os
import cv2
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

class PlateDetector:
   def __init__(self, api_key, confidence=0.8):
       self.confidence = confidence
       self.custom_configuration = InferenceConfiguration(confidence_threshold=confidence)
       self.client = InferenceHTTPClient(
           api_url="https://detect.roboflow.com",
           api_key=api_key
       )
       self.original_dir = "output/original"  
       self.detected_dir = "output/detected"  
       
   def detect_plate(self, frame):
       temp_path = "temp_frame.jpg"
       cv2.imwrite(temp_path, frame)
       try:
            with self.client.use_configuration(self.custom_configuration):
                result = self.client.infer(temp_path, model_id="korea-car-license-plate/1")
            return result['predictions']
       except Exception as e:
           print(f"Error:{e}")
           return []
       finally:
           if os.path.exists(temp_path):
               os.remove(temp_path)
               
   def draw_detections(self, frame, predictions):
       for pred in predictions:
           x = pred['x']
           y = pred['y']
           w = pred['width']
           h = pred['height']
           x1 = int(x - w/2)
           y1 = int(y - h/2)
           x2 = int(x + w/2)
           y2 = int(y + h/2)
           
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
           confidence = f"{pred['confidence']:.2f}"
           cv2.putText(frame, confidence, (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
       return frame

   def process_video(self, video_path, frame_interval=3):
    if not os.path.exists(self.original_dir):
           os.makedirs(self.original_dir)
    if not os.path.exists(self.detected_dir):
           os.makedirs(self.detected_dir)
           
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다.")
        return
        
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
            
        predictions = self.detect_plate(frame)
        
        if predictions:
            filename = f"{base_name}_{saved_count}.jpg"
            original = os.path.join(self.original_dir, filename)
            detected = os.path.join(self.detected_dir, filename)
            cv2.imwrite(original, frame)
            detected_frame = self.draw_detections(frame, predictions)
            cv2.imwrite(detected, detected_frame)
            
            print(f"번호판 감지: {filename} 저장됨")
            saved_count += 1
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

