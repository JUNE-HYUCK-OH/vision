import cv2
import numpy as np

img = cv2.imread('data/imageTextR.png')
if img is None:
    print('이미지를 불러올 수 없습니다.')
else:
    # 좌상, 좌하, 우하 좌표
    pts1 = np.float32([[4, 73], [42, 314], [549, 239]])
    # 변환 후 위치 (직사각형: 좌상, 좌하, 우하)
    width, height = 600, 400
    pts2 = np.float32([[0, 0], [0, height-1], [width-1, height-1]])
    # 어파인 변환 행렬 계산
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (width, height))
    cv2.imshow('Original', img)
    cv2.imshow('Affine Transform', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()