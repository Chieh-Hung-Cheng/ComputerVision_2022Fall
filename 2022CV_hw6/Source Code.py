import cv2
import numpy as np
import matplotlib.pyplot as plt



class Lena():
    def __init__(self):
        self.pic = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
        self.binary = (self.pic >= 128).astype(int)
        self.binary_ds = np.zeros((64, 64)).astype(int)
        self.answer = np.zeros((64, 64)).astype(int)
        self.downsample()
        self.yokoi()

    def showImg(self, image):
        cv2.imshow('IMAGE', image)
        cv2.waitKey()

    def printImg(self, array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j]!=0: print(array[i][j], end='')
                else: print('_', end='')
            print()
        print()

    def downsample(self):
        for i in range(64):
            for j in range(64):
                self.binary_ds[i][j] = self.binary[i*8][j*8]
        self.printImg(self.binary_ds)


    def funH(self,b, c, d, e):
        if b == c and (d!=b or e!=b): return 'q'
        if b==c and (d==b and e==b): return 'r'
        if b!=c: return 's'

    def funF(self, a1, a2, a3, a4):
        if a1=='r' and a2=='r' and a3=='r' and a4=='r':
            return '5'
        else:
            ret_val = 0
            if a1 == 'q': ret_val += 1
            if a2 == 'q': ret_val += 1
            if a3 == 'q': ret_val += 1
            if a4 == 'q': ret_val += 1
            return ret_val

    def inRange(self, x, y):
        if x >= 64 or x<0:
            return False
        if y >= 64 or y<0:
            return False
        return True

    def yokoi(self):
        for i in range(64):
            for j in range(64):
                if self.binary_ds[i][j] == 1:
                    x0 = self.binary_ds[i][j]
                    x1 = 0 if not self.inRange(i + 0, j + 1) else self.binary_ds[i + 0][j + 1]
                    x2 = 0 if not self.inRange(i - 1, j + 0) else self.binary_ds[i - 1][j + 0]
                    x3 = 0 if not self.inRange(i + 0, j - 1) else self.binary_ds[i + 0][j - 1]
                    x4 = 0 if not self.inRange(i + 1, j + 0) else self.binary_ds[i + 1][j + 0]
                    x5 = 0 if not self.inRange(i + 1, j + 1) else self.binary_ds[i + 1][j + 1]
                    x6 = 0 if not self.inRange(i - 1, j + 1) else self.binary_ds[i - 1][j + 1]
                    x7 = 0 if not self.inRange(i - 1, j - 1) else self.binary_ds[i - 1][j - 1]
                    x8 = 0 if not self.inRange(i + 1, j - 1) else self.binary_ds[i + 1][j - 1]

                    # Upper right: [0,0], [0, 1], [-1, 1], [-1, 0]
                    a1 = self.funH(x0, x1, x6, x2)
                    # Upper left: [0,0], [-1, 0], [-1, -1], [0, -1]
                    a2 = self.funH(x0, x2, x7, x3)
                    # Bottom left: [0,0], [0, -1], [1, -1], [1, 0]
                    a3 = self.funH(x0, x3, x8, x4)
                    # Bottom right: [0,0], [1, 0], [1, 1], [0, 1]
                    a4 = self.funH(x0, x4, x5, x1)
                    self.answer[i][j] = self.funF(a1, a2, a3, a4)
                else: self.answer[i][j] = 0
        self.printImg(self.answer)












if __name__ == '__main__':
    lena = Lena()