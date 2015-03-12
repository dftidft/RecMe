# coding=gbk
import cv2
import numpy as np

from ransac import *
from colorFilter import *
from shrink import *


def intersection(a0, a1, b0, b1):
    xlt = (b0 - a0)/(a1 - b1)
    ylt = a0 + a1 * xlt 
    return xlt, ylt


def calibrate(img, warpLt, warpRb):
    mask = colorFilter(img)
    gray = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 500, 0.05, 10, None, mask)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    # for debug only
    '''
    for i in range(corners.shape[0]):
        pt = (corners[i][0, 0], corners[i][0, 1])
        cv2.circle(img, pt, 2, cv2.cv.RGB(0, 0, 255), 2)

    cv2.imshow('mask', mask)
    cv2.imshow('', img)
    cv2.waitKey()
    '''

    candidates = []
    for i in range(corners.shape[0]):
        candidates.append((corners[i][0, 0], corners[i][0, 1]))

    # empirical
    nCandi = 20

    # sort candidates by x
    candidates.sort(key=lambda x:x[0])
    # left line
    (a0, a1) = ransac(candidates[0: nCandi], t=100, d=8)
    # right line
    (b0, b1) = ransac(candidates[-nCandi: -1] + [candidates[-1]], t=100, d=8)
    # sort candidates by y
    candidates.sort(key=lambda x:x[1])
    # top line
    (c0, c1) = ransac(candidates[0: nCandi], t=1, d=8)
    # bottom line
    (d0, d1) = ransac(candidates[-nCandi: -1] + [candidates[-1]], t=1, d=8)

    # print a0, a1, b0, b1, c0, c1, d0, d1

    lt = intersection(a0, a1, c0, c1)
    lb = intersection(a0, a1, d0, d1)
    rt = intersection(b0, b1, c0, c1)
    rb = intersection(b0, b1, d0, d1)

    # for debug only
    '''
    cv2.circle(img, (int(lt[0]), int(lt[1])), 2, cv2.cv.RGB(255, 0, 0), 2)
    cv2.circle(img, (int(lb[0]), int(lb[1])), 2, cv2.cv.RGB(255, 0, 0), 2)
    cv2.circle(img, (int(rt[0]), int(rt[1])), 2, cv2.cv.RGB(255, 0, 0), 2)
    cv2.circle(img, (int(rb[0]), int(rb[1])), 2, cv2.cv.RGB(255, 0, 0), 2)
    print lt, lb, rt, rb
    cv2.imshow('', img)
    cv2.waitKey()
    '''

    srcPts = np.array([lt, lb, rt, rb])
    # print 'lt', lt
    desPts = np.array([(warpLt[0], warpLt[1]), (warpLt[0], warpRb[1]), (warpRb[0], warpLt[1]), (warpRb[0], warpRb[1])])

    (homography, dummy) = cv2.findHomography(srcPts, desPts, cv2.cv.CV_LMEDS)
    return homography


if __name__ == '__main__':

    #img = cv2.imread('g:/dataset/gochessboard/chessboard.jpg')
    img = cv2.imread('G:/Dataset/gochessboard/test1/00100.jpg')
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    homography = calibrate(img, (24.0, 24.0), (456.0, 456.0))
    print homography

    warpedImg = cv2.warpPerspective(img, homography, (480, 480))

    cv2.imshow('', warpedImg)
    cv2.waitKey()


