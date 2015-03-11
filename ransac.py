# coding=gbk

from random import *
from math import *

class LeastSquare:
    def __init__(self):
        self.b0 = 0
        self.b1 = 0

    '''
    def err(self, pts):
        err = 0
        for i in range(len(pts)):
            x = pts[i][0]
            y = pts[i][1]
            yf = self.b0 + self.b1 * x
            err = err + (y - yf) * (y - yf)
        return err
    '''

    # 点到直线的距离
    def err(self, pts):
        err = 0
        for i in range(len(pts)):
            x = pts[i][0]
            y = pts[i][1]
            err = err + abs(self.b0 + self.b1 * x - y)
        return err / sqrt(self.b0 ** 2 + self.b1 ** 2)

    def fit(self, pts):
        n = len(pts)
        xm = 0  #mean x
        ym = 0  #mean y
        for i in range(len(pts)):
            xm = xm + pts[i][0]
            ym = ym + pts[i][1]
        xm = xm / n
        ym = ym / n
        s1 = 0
        s2 = 0
        for i in range(n):
            x = pts[i][0]
            y = pts[i][1]
            s1 = s1 + x * y
            s2 = s2 + x * x
        self.b1 = (s1 - n * xm * ym) / (s2 - n * xm * xm)
        self.b0 = ym - self.b1 * xm
        return (self.b0, self.b1)

'''
Ransac 算法
# n - 适用于模型的最小数据数（直线，此处为2）
# k - 算法的迭代次数
# t - 数据适用于模型的阈值
# d - 模型适用于数据集的最小数据个数   
''' 
def ransac(pts, n = 2, k = 100, t = 1, d = 20):
    bestModel = None
    bestErr = 10000000
    lsq = LeastSquare()
    for iter in range(k):
        # get n samples from dataset
        maybeInlier = []
        for i in range(n):
            maybeInlier.append(randint(0, len(pts) - 1))
        # estimate model by n sampled data
        x0 = pts[maybeInlier[0]][0]
        x1 = pts[maybeInlier[1]][0]
        y0 = pts[maybeInlier[0]][1]
        y1 = pts[maybeInlier[1]][1]
        div = x1 - x0
        if div == 0:
            div = 0.00001
        b0 = ((x1 * y0) - (x0 * y1)) / div
        b1 = (y1 - y0) / div
        alsoInlier = []
        for i in range(len(pts)):
            if i not in maybeInlier:
                yi = b0 + b1 * pts[i][0]
                if abs(yi - pts[i][1]) < t:
                    alsoInlier.append(i)
        if (len(alsoInlier) > d):
            # estimate better model
            maybeInlier = maybeInlier + alsoInlier
            tmpPts = []
            for i in range(len(maybeInlier)):
                tmpPts.append(pts[maybeInlier[i]])
                tmpPts.append(pts[maybeInlier[i]])
            (b0, b1) = lsq.fit(tmpPts)
            err = lsq.err(tmpPts)
            if err < bestErr:
                bestErr = err
                bestModel = (b0, b1)
    # print bestErr
    return bestModel

        

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # create test data
    x = range(0, 100)
    y = []
    for xi in x:
      y.append(2 * xi + 3 + gauss(0, 1))

    for i in range(10):
      y[i * 5] = 200

    pts = []
    for i in range(len(x)):
        pts.append((x[i], y[i]))
        
    plt.plot(x, y, 'g.')
    
    lsq = LeastSquare()
    (b0, b1) = lsq.fit(pts)
    x2 = range(0, 100)
    y2 = []
    for xi in x2:
        y2.append(b1 * xi + b0)
    plt.plot(x2, y2, 'b')
    
    (b0, b1) = ransac(pts, 2, 100, 1, 20)
    x3 = range(0, 100)
    y3 = []
    for xi in x3:
        y3.append(b1 * xi + b0)
    plt.plot(x3, y3, 'r')

    plt.show()
