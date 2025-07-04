from ultralytics import YOLO
import matplotlib.pyplot as plt

# 유튜브 영상 링크
video_url = "https://www.youtube.com/watch?v=x-ovfT7Tt8s"

# YOLOv8 모델 로드
model = YOLO("yolov8n.pt")

# BoT-SORT 트래커로 자동차 트래킹
# stream=True로 프레임별 결과를 반복자 형태로 받음
results = model.track(
    source=video_url,
    show=True,
    tracker="botsort.yaml",
    persist=True,
    stream=True,
    classes=[2, 5, 7]  # car, bus, truck
)

unique_ids = set()
car_counts = []

for r in results:
    # track id가 있는 box만 추출
    if hasattr(r.boxes, "id") and r.boxes.id is not None:
        ids = r.boxes.id.int().cpu().tolist()
        # 자동차 클래스만 필터링
        car_cls = r.boxes.cls.int().cpu().tolist()
        for i, cls in enumerate(car_cls):
            if cls in [2, 5, 7]:
                unique_ids.add(ids[i])
    car_counts.append(len(unique_ids))
    # ESC(q)로 종료
    import cv2
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

# 그래프 출력
plt.figure(figsize=(10,4))
plt.plot(car_counts, label='Unique Car Tracks')
plt.xlabel('Frame')
plt.ylabel('Unique Car Count')
plt.title('YOLOv8 + BoT-SORT Car Count Over Time')
plt.legend()
plt.show()