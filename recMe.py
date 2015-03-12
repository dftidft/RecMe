# coding=gbk

import cv2
import numpy as np
from detectPiece import *


def checkDetectStatus(action):
    actionPast = False
    actionNow = False
    mid = len(action) / 2
    for i in range(mid):
        if action[i]:
            actionPast = True
            break
    for i in range(mid, len(action)):
        if action[i]:
            actionNow = True
            break
    if actionPast and not actionNow:
        return True
    else:
        return False

seqDir = 'data'
startFrame = 120
endFrame = 1654

bgsub = cv2.BackgroundSubtractorMOG2()
kernel = np.ones((7, 7), np.uint8)
fgmask = []

# 记录一段时间的动作，避免动作停滞时不能检出
nActionFrames = 6
actionInLastFrames = [False for i in range(nActionFrames)]

for iFrame in range(startFrame, endFrame):

    img = cv2.imread('%s/%05d.jpg' % (seqDir, iFrame))
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)

    fgmask = bgsub.apply(img, None, 0.01)
    fgmask = cv2.morphologyEx(fgmask, cv2.cv.CV_MOP_OPEN, kernel, 1)
    fgmask = cv2.morphologyEx(fgmask, cv2.cv.CV_MOP_CLOSE, kernel, 1)

    if iFrame == startFrame:
        homography = calibrate(img, (24.0, 24.0), (456.0, 456.0))
        sim = cv2.warpPerspective(img, homography, (480, 480))
    else:
        if np.sum(np.sum(fgmask)) / 255 < 10:
            actionInLastFrames = actionInLastFrames[1:] + [False]
            if checkDetectStatus(actionInLastFrames):
                # good frame for detecting chess pieces

                warpedImg = cv2.warpPerspective(img, homography, (480, 480))
                slots, slotsPos = detectPiece(warpedImg, 24, 456)
                # sim = np.ones((480, 480, 3), dtype=np.uint8) * 100
                for i in range(len(slots)):
                    if slots[i] == 1:
                        cv2.circle(warpedImg, (int(slotsPos[i][0]), int(slotsPos[i][1])), 5, cv2.cv.RGB(255, 0, 0), 2)
                    elif slots[i] == 2:
                        cv2.circle(warpedImg, (int(slotsPos[i][0]), int(slotsPos[i][1])), 5, cv2.cv.RGB(0, 0, 255), 2)
                sim = warpedImg
        else:
            actionInLastFrames = actionInLastFrames[1:] + [True]

    cv2.imshow('img', img)
    # cv2.imshow('mask', fgmask)
    cv2.imshow('sim', sim)
    if cv2.waitKey(30) == 27:
        break