# coding=gbk

import cv2
import numpy as np
from colorFilter import *
from calibrate import *
import sklearn.cluster


# 参数
#       left 棋盘左边(上边)位置
#       right 棋盘右边(下边)位置
# 返回
#       19*19 = 361 的列表slots，其中slot[i] = 0表示无棋子， 1表示黑子， 0表示白子
#       19*19 = 361 的列表位置slotsPos，每个位置包括slot[i]中心的(x, y)坐标

def detectPieceTrain(img, left, right):

    # 采样面积
    radius = 6

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
        scores.append([s])
        avg += s
    avg /= len(slotsPos)

    scores = np.array(scores, dtype=float)
    kMeans = sklearn.cluster.KMeans(n_clusters=3, max_iter=10, tol=0.1)
    kMeans.fit(scores)
    labels = kMeans.predict(scores)

    # print labels
    print kMeans.cluster_centers_

    idx = np.argsort(kMeans.cluster_centers_.ravel())

    # define blank = 0, black = 1, white = 2
    slots = []
    for l in labels:
        if l == idx[0]:
            slots.append(1)
        elif l == idx[1]:
            slots.append(0)
        else:
            slots.append(2)
    return slots, slotsPos


if __name__ == '__main__':

    img = cv2.imread('G:/Dataset/gochessboard/test1/01600.jpg')
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    homography = calibrate(img, (24.0, 24.0), (456.0, 456.0))

    warpedImg = cv2.warpPerspective(img, homography, (480, 480))
    slots, slotsPos = detectPieceTrain(warpedImg, 24, 456)
    # sim = np.ones((480, 480, 3), dtype=np.uint8) * 100
    for i in range(len(slots)):
        if slots[i] == 1:
            cv2.circle(warpedImg, (int(slotsPos[i][0]), int(slotsPos[i][1])), 5, cv2.cv.RGB(255, 0, 0), 2)
        elif slots[i] == 2:
            cv2.circle(warpedImg, (int(slotsPos[i][0]), int(slotsPos[i][1])), 5, cv2.cv.RGB(0, 0, 255), 2)

    cv2.imshow('', warpedImg)
    cv2.waitKey()