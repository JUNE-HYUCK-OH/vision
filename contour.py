import cv2
import numpy as np

img = cv2.imread('data/opencv-logo.png')
if img is None:
    print('이미지를 불러올 수 없습니다.')
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #임계값 적용(이진화)
    ret, thresh = cv2.threshold(gray, 70, 255, 0)
    #등고선 찾기
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 등고선 그리기
    img_contour = img.copy()
    cv2.drawContours(img_contour, contours, -1, (255, 255, 255), 2)
    print(f'등고선 개수: {len(contours)}')
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        print(f'등고선 {i+1} 넓이: {area}')
    cv2.imshow('Contours', img_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()