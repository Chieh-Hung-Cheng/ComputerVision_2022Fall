import cv2
import numpy as np
import matplotlib.pyplot as plt



class Lena():
    def __init__(self):
        self.pic = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
        self.binary = (self.pic >= 128).astype(int)
        self.binary_ds = np.zeros((64, 64)).astype(int)
        self.yk_result = np.zeros((64, 64)).astype(int)
        self.pr_result = np.zeros((64, 64)).astype(str)
        self.shr_result = np.zeros((64, 64)).astype(int)
        self.cnt = 1
        self.iter = 0
        self.execute()

    def execute(self):
        self.downsample()

        while(self.cnt!=0):
            print('Iter:{}'.format(self.iter))
            self.iter +=1
            self.cnt = 0
            self.yokoi()
            self.pairRelationship()
            self.connectedShrink()

            self.binary_ds = self.shr_result
            self.yk_result = np.zeros((64, 64)).astype(int)
            self.pr_result = np.zeros((64, 64)).astype(str)
            self.shr_result = np.zeros((64, 64)).astype(int)
        self.showImg(self.binary_ds*255, save=True)


    def showImg(self, image, save=False):
        cv2.imshow('IMAGE', image.astype(np.uint8))
        cv2.waitKey()
        if save: cv2.imwrite('thinned.bmp',image.astype(np.uint8))

    def printImg(self, array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j]!= 0 and array[i][j] != '0': print(array[i][j], end='')
                else: print('_', end='')
            print()
        print()

    def downsample(self):
        for i in range(64):
            for j in range(64):
                self.binary_ds[i][j] = self.binary[i*8][j*8]
        self.printImg(self.binary_ds)

    def inRange(self, x, y):
        if x >= 64 or x<0:
            return False
        if y >= 64 or y<0:
            return False
        return True

    def yokoi(self):
        def funH(b, c, d, e):
            if b == c and (d != b or e != b): return 'q'
            if b == c and (d == b and e == b): return 'r'
            if b != c: return 's'

        def funF(a1, a2, a3, a4):
            if a1 == 'r' and a2 == 'r' and a3 == 'r' and a4 == 'r':
                return '5'
            else:
                ret_val = 0
                if a1 == 'q': ret_val += 1
                if a2 == 'q': ret_val += 1
                if a3 == 'q': ret_val += 1
                if a4 == 'q': ret_val += 1
                return ret_val

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
                    a1 = funH(x0, x1, x6, x2)
                    # Upper left: [0,0], [-1, 0], [-1, -1], [0, -1]
                    a2 = funH(x0, x2, x7, x3)
                    # Bottom left: [0,0], [0, -1], [1, -1], [1, 0]
                    a3 = funH(x0, x3, x8, x4)
                    # Bottom right: [0,0], [1, 0], [1, 1], [0, 1]
                    a4 = funH(x0, x4, x5, x1)
                    self.yk_result[i][j] = funF(a1, a2, a3, a4)
                else: self.yk_result[i][j] = 0
        self.printImg(self.yk_result)

    def pairRelationship(self):
        def funH(a, m):
            if a==m: return 1
            else: return 0

        def funF(a1, a2, a3, a4):
            if (a1 + a2 + a3 + a4) < 1: return 'q'
            else: return 'p'

        for i in range(64):
            for j in range(64):
                if self.yk_result[i][j]==1:

                    if  i==14 and j == 16: pass
                    x0 = self.yk_result[i][j]
                    x1 = 0 if not self.inRange(i + 0, j + 1) else self.yk_result[i + 0][j + 1]
                    x2 = 0 if not self.inRange(i - 1, j + 0) else self.yk_result[i - 1][j + 0]
                    x3 = 0 if not self.inRange(i + 0, j - 1) else self.yk_result[i + 0][j - 1]
                    x4 = 0 if not self.inRange(i + 1, j + 0) else self.yk_result[i + 1][j + 0]

                    a1 = funH(x0, x1)
                    a2 = funH(x0, x2)
                    a3 = funH(x0, x3)
                    a4 = funH(x0, x4)

                    self.pr_result[i][j] = funF(a1, a2, a3, a4)
                elif self.yk_result[i][j]==0: self.pr_result[i][j] = '0'
                else: self.pr_result[i][j] = 'q'
        self.printImg(self.pr_result)

    def connectedShrink(self):
        def funH(b,c,d,e):
            if b == c and (d!=b or e!=b): return 1
            else: return 0
        def funF(a1,a2,a3,a4):
            if a1+a2+a3+a4==1: # make it background
                self.cnt += 1
                return 0
            else:
                return 1

        for i in range(64):
            for j in range(64):
                if self.pr_result[i][j] == 'p':
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
                    a1 = funH(x0, x1, x6, x2)
                    # Upper left: [0,0], [-1, 0], [-1, -1], [0, -1]
                    a2 = funH(x0, x2, x7, x3)
                    # Bottom left: [0,0], [0, -1], [1, -1], [1, 0]
                    a3 = funH(x0, x3, x8, x4)
                    # Bottom right: [0,0], [1, 0], [1, 1], [0, 1]
                    a4 = funH(x0, x4, x5, x1)
                    self.shr_result[i][j] = funF(a1, a2, a3, a4)
                elif self.pr_result[i][j] == 'q':
                    self.shr_result[i][j] = 1
                else:
                    self.shr_result[i][j] = 0
        self.printImg(self.shr_result)

if __name__ == '__main__':
    lena = Lena()