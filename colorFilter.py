# coding=gbk
import cv2
from shrink import *


def colorFilter(img):
    # yellow hsv > 60, s > 80, 220 > v > 50
    img = cv2.blur(img, (7, 7))
    hsv = cv2.cvtColor(img, cv2.cv.CV_BGR2HSV)
    #lower_yellow = np.array([10, 50, 50], dtype=np.uint8)
    #upper_yellow = np.array([50, 255, 255], dtype=np.uint8)
    lower_yellow = np.array([0, 0, 0], dtype=np.uint8)
    upper_yellow = np.array([60, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((7, 7))
    mask = cv2.morphologyEx(mask, cv2.cv.CV_MOP_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.cv.CV_MOP_CLOSE, kernel, 1)
    return mask


if __name__ == '__main__':

    #img = cv2.imread('g:/dataset/gochessboard/chessboard.jpg')
    img = cv2.imread('G:/Dataset/gochessboard/test1/00001.jpg')
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    size = img.shape
    # yellow hsv > 60, s > 80, 220 > v > 50
    mask = colorFilter(img)
    #mask = shrink(mask, 10)
    cv2.imshow('mask', mask)
    cv2.waitKey()

    gray = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 500, 0.05, 10, None, mask)

    for i in range(corners.shape[0]):
        pt = (corners[i][0, 0], corners[i][0, 1])
        cv2.circle(img, pt, 2, cv2.cv.RGB(0, 0, 255), 2)

    cv2.imshow('mask', mask)
    cv2.imshow('', img)
    cv2.waitKey()

