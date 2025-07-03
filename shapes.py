import cv2
import numpy as np

img = cv2.imread('data/shapes.jpg')
if img is None:
    print('이미지를 불러올 수 없습니다.')
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    result = img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # 너무 작은 노이즈 무시
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / h
        if len(approx) > 12:
            # 꼭지점이 많으면 원으로 인식
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(result, (int(cx), int(cy)), int(radius), (0, 255, 255), 2)
            cv2.putText(result, 'Circle', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        elif len(approx) == 4:
            # 사각형 계열
            if abs(aspect - 1) < 0.1:
                cv2.putText(result, 'Square', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                cv2.rectangle(result, (x, y), (x+w, y+h), (255,0,0), 2)
            else:
                cv2.putText(result, 'Rectangle', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.rectangle(result, (x, y), (x+w, y+h), (0,255,0), 2)
        else:
            # 기타 도형(삼각형 등)
            cv2.drawContours(result, [approx], -1, (0,0,255), 2)
            cv2.putText(result, f'Poly({len(approx)})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.imshow('Detected Shapes', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()