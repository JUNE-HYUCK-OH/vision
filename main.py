import cv2 # OpenCV 로드
cap = cv2.VideoCapture(0)  # 웹캠에서 비디오 객체 cap 생성

if not cap.isOpened():
    print("동영상을 불러올 수 없습니다.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 프레임을 흑백으로 변환
        cv2.imshow("video", gray)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC 키로 종료
            break
    cap.release()
    cv2.destroyAllWindows()