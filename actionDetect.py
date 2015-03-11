import cv2
import numpy as np

seqDir = 'g:/Dataset/gochessboard/test1'
startFrame = 120
endFrame = 1654

bgsub = cv2.BackgroundSubtractorMOG2()
kernel = np.ones((7, 7), np.uint8)
fgmask = []

actionInLastFrame = False
actionInCurrentFrame = False

for iFrame in range(startFrame, endFrame):
    img = cv2.imread('%s/%05d.jpg' % (seqDir, iFrame))
    fgmask = bgsub.apply(img, None, 0.01)
    fgmask = cv2.morphologyEx(fgmask, cv2.cv.CV_MOP_OPEN, kernel, 1)
    fgmask = cv2.morphologyEx(fgmask, cv2.cv.CV_MOP_CLOSE, kernel, 1)
    if np.sum(np.sum(fgmask)) / 255 < 10:
        if actionInLastFrame:
            # good frame for detecting chess pieces
            actionInLastFrame = False
    else:
        actionInLastFrame = True
    fgmask = cv2.resize(fgmask, dsize=None, fx=0.5, fy=0.5)
    cv2.imshow('img', img)
    cv2.imshow('mask', fgmask)
    if cv2.waitKey(30) == 27:
        break
