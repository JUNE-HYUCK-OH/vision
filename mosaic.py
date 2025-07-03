import cv2

img = cv2.imread('data/Dog.jpg')
if img is None:
    print('이미지를 불러올 수 없습니다.')
else:
    # 좌상단 (293, 67), 우하단 (566, 386) 영역 모자이크 처리
    x1, y1 = 293, 67
    x2, y2 = 566, 386
    roi = img[y1:y2, x1:x2]
    mosaic = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(mosaic, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = mosaic
    cv2.imshow('Mosaic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()