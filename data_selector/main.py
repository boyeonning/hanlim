import os
import glob
from app.plate_detector import PlateDetector

def main():
   api_key = "hqYhmIIV0wGPFJWKoebi"
   dir_path = '/Users/boyeon/workspace/24-hallim/data'
   video_files = sorted(glob.glob(os.path.join(dir_path, '**', '*.avi'), recursive=True))

   detector = PlateDetector(api_key)
   for video_path in video_files:
       print(f"처리 중인 비디오: {video_path}")
       detector.process_video(video_path)

if __name__ == "__main__":
    main()