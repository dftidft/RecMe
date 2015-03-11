# coding=gbk

# shrink mask
# ȥ��mask�ܱ�margin������������ݣ���Сmask�ķ�Χ

import numpy as np


def removeMargin(mask, dim, margin):
    size = mask.shape
    start = np.sum(np.cumsum(mask, dim) == 0, dim)
    # print start
    end = start + margin
    for i in range(end.shape[0]):
        if end[i] > size[dim]:
            end[i] = size[dim]
    # print end
    if dim == 1:
        for i in range(end.shape[0]):
            mask[i, start[i]: end[i]] = 0
    else:
        for i in range(end.shape[0]):
            mask[start[i]: end[i], i] = 0


def shrink(mask, margin):
    removeMargin(mask, 1, margin)
    mask = mask[:, ::-1]
    removeMargin(mask, 1, margin)
    mask = mask[:, ::-1]
    removeMargin(mask, 0, margin)
    mask = mask[::-1, :]
    removeMargin(mask, 0, margin)
    mask = mask[::-1, :]
    return mask
