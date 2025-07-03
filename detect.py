import cv2

# haarcascade 파일 경로
cascade_path = 'data/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

img = cv2.imread('data/lena.jpg')
if img is None:
    print('이미지를 불러올 수 없습니다.')
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    print(f'인식된 얼굴 개수: {len(faces)}')
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()