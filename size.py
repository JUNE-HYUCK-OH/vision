import cv2

img = cv2.imread('data/ear.jpg')
if img is None:
    print('이미지를 불러올 수 없습니다.')
else:
    # 1. 2배 확대
    img_up = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('2x', img_up)
    # 2. 0.5배 축소
    img_down = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('0.5x', img_down)
    # 3. 300x300으로 강제 변환
    img_300 = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('300x300', img_300)
    # 4. 100x200으로 강제 변환
    img_100_200 = cv2.resize(img, (100, 200), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('100x200', img_100_200)
    # 5. 원본
    cv2.imshow('original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()