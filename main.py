import cv2 # OpenCV 로드
cap = cv2.VideoCapture("data/Megamind.avi")
# 이 파일과 같은 경로에 있는 data 폴더
# 안에 Megamind.avi를 읽어서  openCV 객체
# img로 저장
if not cap.isOpened():
    print("동영상을 불러올 수 없습니다.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("video", frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC 키로 종료
            break
    cap.release()
    cv2.destroyAllWindows()