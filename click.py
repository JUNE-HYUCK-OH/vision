import cv2
import numpy as np

img = cv2.imread('data/earphone.jpg')
if img is None:
    print('이미지를 불러올 수 없습니다.')
else:
    # 창에 맞게 이미지 크기 조정 (예: 최대 800x600)
    max_width, max_height = 800, 600
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    # 원본 이미지의 네 점 (좌상, 좌하, 우하, 우상)
    pts1 = np.float32([[61, 300], [193, 480], [384, 347], [246, 152]])
    # 변환 후 위치 (직사각형)
    width, height = 300, 400
    pts2 = np.float32([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]])
    # 투시 변환 행렬 계산
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (width, height))
    cv2.imshow('Original', img)
    cv2.imshow('Projective Transform', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()