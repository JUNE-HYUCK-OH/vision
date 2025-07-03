import cv2
import numpy as np
import datetime

# haarcascade 파일 경로
cascade_path = 'data/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 얼굴 인식
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # 오른쪽 위에 날짜와 시간 표시
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        (w_text, h_text), _ = cv2.getTextSize(now, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, now, (frame.shape[1] - w_text - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, now, (frame.shape[1] - w_text - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
        # 왼쪽 위에 녹화 표시
        cv2.circle(frame, (20, 25), 10, (0,0,255), -1)
        cv2.putText(frame, 'REC', (40, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC 키로 종료
            break
    cap.release()
    cv2.destroyAllWindows()