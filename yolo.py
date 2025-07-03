from ultralytics import YOLO
import cv2
import yt_dlp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

url = 'https://www.youtube.com/watch?v=x-ovfT7Tt8s'

# yt-dlp로 유튜브 영상의 direct stream url 얻기
ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    stream_url = info['url']

cap = cv2.VideoCapture(stream_url)
model = YOLO('yolov8n.pt')
car_classes = [2, 3, 5, 7]

# 간단한 트래킹: 중심점 좌표를 저장하여 중복 카운트 방지
prev_centers = []
car_id = 0
car_tracks = {}
car_count = 0
frame_counts = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    boxes = results[0].boxes
    centers = []
    for i, (cls, box) in enumerate(zip(boxes.cls, boxes.xyxy)):
        if int(cls) in car_classes:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centers.append((cx, cy))
            # 기존 트랙과 비교해 새로운 차인지 확인
            found = False
            for pid, track in car_tracks.items():
                if np.linalg.norm(np.array(track[-1]) - np.array([cx, cy])) < 50:
                    car_tracks[pid].append((cx, cy))
                    found = True
                    break
            if not found:
                car_tracks[car_id] = deque([(cx, cy)], maxlen=30)
                car_id += 1
    # 프레임마다 현재 트랙 수를 기록
    frame_counts.append(len(car_tracks))
    # 화면에 카운트 표시
    cv2.putText(annotated_frame, f"Unique Car Tracks: {len(car_tracks)}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('YOLOv8 Car Counting', annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# 그래프 그리기
plt.figure(figsize=(10,4))
plt.plot(frame_counts, label='Unique Car Tracks')
plt.xlabel('Frame')
plt.ylabel('Car Count')
plt.title('YOLOv8 Car Count Over Time')
plt.legend()
plt.show()