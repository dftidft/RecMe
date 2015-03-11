import cv2
import numpy as np
from colorFilter import *
from calibrate import *
import sklearn.cluster

img = cv2.imread('G:/Dataset/gochessboard/test1/01600.jpg')
img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
homography = calibrate(img, (24.0, 24.0), (456.0, 456.0))

warpedImg = cv2.warpPerspective(img, homography, (480, 480))
gray = cv2.cvtColor(warpedImg, cv2.cv.CV_BGR2GRAY)

lines = np.linspace(24, 456, 19)
print len(lines)
# print lines

slotsPos = [(x, y) for x in lines for y in lines]
print slotsPos
print len(slotsPos)

scores = []
avg = 0
for pts in slotsPos:
    patch = gray[pts[1] - 6: pts[1] + 6, pts[0] - 6: pts[0] + 6]
    s = np.sum(np.sum(patch))
    scores.append([s])
    avg += s
avg /= len(slotsPos)

scores = np.array(scores, dtype=float)
print scores.shape
kMeans = sklearn.cluster.KMeans(n_clusters=3, max_iter=10, tol=0.1)
kMeans.fit(scores)
labels = kMeans.predict(scores)

print labels
print kMeans.cluster_centers_

idx = np.argsort(kMeans.cluster_centers_.ravel())
print idx

# define blank = 0, black = 1, white = 2
slots = []
for l in labels:
    if l == idx[0]:
        slots.append(1)
    elif l == idx[1]:
        slots.append(0)
    else:
        slots.append(2)


#sim = np.ones((480, 480, 3), dtype=np.uint8) * 100
for i in range(len(slots)):
    if slots[i] == 1:
        cv2.circle(warpedImg, (int(slotsPos[i][0]), int(slotsPos[i][1])), 5, cv2.cv.RGB(255, 0, 0), 2)
    elif slots[i] == 2:
        cv2.circle(warpedImg, (int(slotsPos[i][0]), int(slotsPos[i][1])), 5, cv2.cv.RGB(0, 0, 255), 2)

cv2.imshow('', warpedImg)
cv2.waitKey()