import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


class Kernel():
    class KernelUnit():
        row = 0
        col = 0
        weight = 0

        def __init__(self, r, c, w):
            self.row = r
            self.col = c
            self.weight = w

    def __init__(self, *args, norm=1.0):
        self.length = int(np.sqrt(len(args)))
        self.d = int((self.length - 1) / 2)
        self.elements = []
        self.normalize = norm
        for i in range(self.length):
            for j in range(self.length):
                self.elements.append(self.KernelUnit(i - self.d, j - self.d, args[i * self.length + j]))

    def DoKernelCov(self, img, row, col):
        retVal = 0
        for elm in self.elements:
            retVal += (elm.weight * img[row + elm.row][col + elm.col])
        retVal *= self.normalize
        return retVal


class Lena():
    def __init__(self):
        self.pic = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
        self.Execute()

    def Execute(self):
        laplacianMask = Kernel(0,1,0,1,-4,1,0,1,0)
        laplacian = self.DetectZeroCrossingEdge(self.pic, laplacianMask, 15)
        self.ShowSaveImg(laplacian, save=True, filename='laplacian15_a')

        laplacianMask2 = Kernel(1,1,1,1,-8,1,1,1,1,norm=1/3)
        laplacian2 = self.DetectZeroCrossingEdge(self.pic, laplacianMask2, 15)
        self.ShowSaveImg(laplacian2, save=True, filename='laplacian15_b')

        miniVariMask = Kernel(2,-1,2,-1,-4,-1,2,-1,2, norm=1/3)
        miniVariLapla = self.DetectZeroCrossingEdge(self.pic, miniVariMask, 20)
        self.ShowSaveImg(miniVariLapla, save=True, filename='minivariancelaplacian20')

        laplaGaussMask = Kernel(0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0,
                                0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0,
                                0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0,
                                -1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1,
                                -1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1,
                                -2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2,
                                -1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1,
                                -1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1,
                                0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0,
                                0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0,
                                0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0)
        laplaGuass = self.DetectZeroCrossingEdge(self.pic, laplaGaussMask, 3000)
        self.ShowSaveImg(laplaGuass, save=True, filename='laplaGuass3000')

        diffGaussMask = Kernel(-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1,
                               -3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3,
                               -4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4,
                               -6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6,
                               -7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7,
                               -8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8,
                               -7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7,
                               -6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6,
                               -4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4,
                               -3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3,
                               -1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1)
        diffGauss = self.DetectZeroCrossingEdge(self.pic, diffGaussMask, 1)
        self.ShowSaveImg(diffGauss, save=True, filename='diffGauss1')

    def ShowSaveImg(self, image, save=False, show=True, filename=None):
        if show:
            cv2.imshow('IMAGE', image.astype(np.uint8))
            cv2.waitKey()
        if save: cv2.imwrite('{}.bmp'.format(filename), image.astype(np.uint8))

    def PrintImg(self, array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j] != 0 and array[i][j] != '0':
                    print(array[i][j], end='')
                else:
                    print('_', end='')
            print()
        print()

    def ISInRange(self, x, y):
        if x >= 512 or x < 0:
            return False
        if y >= 512 or y < 0:
            return False
        return True

    def DetectZeroCrossingEdge(self, inPic, Kernel, thr):
        laplaPic = np.zeros_like(inPic).astype(int)
        paddedPic = cv2.copyMakeBorder(inPic, Kernel.d, Kernel.d, Kernel.d, Kernel.d, cv2.BORDER_REPLICATE)

        for i in range(inPic.shape[0]):
            for j in range(inPic.shape[1]):
                resultVal = Kernel.DoKernelCov(paddedPic, i + Kernel.d, j + Kernel.d)
                if resultVal >= thr:
                    laplaPic[i][j] = 1
                elif resultVal <= -1 * thr:
                    laplaPic[i][j] = -1
                else:
                    laplaPic[i][j] = 0

        def ISCrossing(img, row, col):
            for i in range(-1 * 1, 1 + 1):
                for j in range(-1 * 1, 1 + 1):
                    if img[row + i][col + j] == -1: return True
            return False

        retPic = np.zeros_like(inPic)
        paddedLaplaPic = cv2.copyMakeBorder(laplaPic, Kernel.d, Kernel.d, Kernel.d, Kernel.d, cv2.BORDER_REPLICATE)
        for i in range(inPic.shape[0]):
            for j in range(inPic.shape[1]):
                if paddedLaplaPic[i + Kernel.d][j + Kernel.d] != 1:
                    retPic[i][j] = 255
                else:
                    if ISCrossing(paddedLaplaPic, i + Kernel.d, j + Kernel.d):
                        retPic[i][j] = 0
                    else:
                        retPic[i][j] = 255
        return retPic


if __name__ == '__main__':
    lena = Lena()
