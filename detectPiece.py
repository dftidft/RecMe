# coding=gbk

import cv2
import numpy as np
from colorFilter import *
from calibrate import *


# 参数
#       left 棋盘左边(上边)位置
#       right 棋盘右边(下边)位置
# 返回
#       19*19 = 361 的列表slots，其中slot[i] = 0表示无棋子， 1表示黑子， 0表示白子
#       19*19 = 361 的列表位置slotsPos，每个位置包括slot[i]中心的(x, y)坐标

def detectPiece(img, left, right):

    # 采样面积
    radius = 5

    gray = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    lines = np.linspace(left, right, 19)
    # print len(lines)
    # print lines

    slotsPos = [(x, y) for x in lines for y in lines]
    # print slotsPos

    scores = []
    avg = 0
    for pts in slotsPos:
        patch = gray[pts[1] - radius: pts[1] + radius, pts[0] - radius: pts[0] + radius]
        s = np.sum(np.sum(patch)) / ((2 * radius + 1)**2)
        scores.append(s)
        avg += s
    avg /= len(slotsPos)

    # 基准亮度
    # blank = 120
    # black = 30
    # white = 160
    # slot_i_value: blank = 0, black = 1, white = 2

    baseLine = [120.0, 30.0, 160.0]
    slots = []

    for s in scores:
        minI = 0
        minDiff = abs(s - baseLine[0])
        for i in range(1, 3):
            if abs(s - baseLine[i]) < minDiff:
                minI = i
                minDiff = abs(s - baseLine[i])
        slots.append(minI)

    return slots, slotsPos


if __name__ == '__main__':

    img0 = cv2.imread('G:/Dataset/gochessboard/test1/00100.jpg')
    img0 = cv2.resize(img0, dsize=None, fx=0.5, fy=0.5)

    img = cv2.imread('G:/Dataset/gochessboard/test1/01300.jpg')
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    homography = calibrate(img0, (24.0, 24.0), (456.0, 456.0))

    warpedImg = cv2.warpPerspective(img, homography, (480, 480))
    slots, slotsPos = detectPiece(warpedImg, 24, 456)
    # sim = np.ones((480, 480, 3), dtype=np.uint8) * 100
    for i in range(len(slots)):
        if slots[i] == 1:
            cv2.circle(warpedImg, (int(slotsPos[i][0]), int(slotsPos[i][1])), 5, cv2.cv.RGB(255, 0, 0), 2)
        elif slots[i] == 2:
            cv2.circle(warpedImg, (int(slotsPos[i][0]), int(slotsPos[i][1])), 5, cv2.cv.RGB(0, 0, 255), 2)

    cv2.imshow('', warpedImg)
    cv2.waitKey()