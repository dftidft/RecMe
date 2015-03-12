# coding=gbk
import cv2

img = cv2.imread('g:/dataset/gochessboard/test1/00001.jpg')

size = img.shape
print size
gray = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 500, 0.05, 10)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

for i in range(corners.shape[0]):
    # print corners[i][0, 0]
    pt = (corners[i][0, 0], corners[i][0, 1])
    cv2.circle(img, pt, 2, cv2.cv.RGB(0, 0, 255), 2)

cv2.imshow('', img)
cv2.waitKey()
