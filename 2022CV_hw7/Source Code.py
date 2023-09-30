import cv2
import numpy as np
import matplotlib.pyplot as plt



class Lena():
    def __init__(self):
        self.pic = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
        self.binary = (self.pic >= 128).astype(int)
        self.binds = self.downsample(self.binary) # BINary Down Sample
        self.cnt = -1
        self.iter = 0
        self.thinning()

    def thinning(self):

        ans = np.copy(self.binds)
        while(self.cnt != 0):
            print('Iteration {}'.format(self.iter))
            # Initialize
            self.cnt = 0

            # Execute
            yk = self.yokoi(ans)
            self.printImg(yk)
            pr = self.pairRelationship(yk)
            self.printImg(pr)
            sh = self.connectedShrink(pr, ans)
            self.printImg(sh)

            # Update
            ans = sh
            self.iter += 1
        self.showImg(ans, save=True)

    def showImg(self, image, save=False):
        image *= 255
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

    def downsample(self, array):
        ret_array = np.zeros((64,64)).astype(int)
        for i in range(64):
            for j in range(64):
                ret_array[i][j] = array[i*8][j*8]
        return ret_array

    def inRange(self, x, y):
        if x >= 64 or x<0:
            return False
        if y >= 64 or y<0:
            return False
        return True

    def yokoi(self, array):
        # Input: Binary down-sampled array(shape 64,64, integer): 0 for background and 1 for foreground
        # Output: Yokoi result(shape 64,64, integer): # indicating the Yokoi connectivity number
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

        ret_array = np.zeros((64,64)).astype(int)

        for i in range(64):
            for j in range(64):
                if array[i][j] == 1:
                    x0 = array[i][j]
                    x1 = 0 if not self.inRange(i + 0, j + 1) else array[i + 0][j + 1]
                    x2 = 0 if not self.inRange(i - 1, j + 0) else array[i - 1][j + 0]
                    x3 = 0 if not self.inRange(i + 0, j - 1) else array[i + 0][j - 1]
                    x4 = 0 if not self.inRange(i + 1, j + 0) else array[i + 1][j + 0]
                    x5 = 0 if not self.inRange(i + 1, j + 1) else array[i + 1][j + 1]
                    x6 = 0 if not self.inRange(i - 1, j + 1) else array[i - 1][j + 1]
                    x7 = 0 if not self.inRange(i - 1, j - 1) else array[i - 1][j - 1]
                    x8 = 0 if not self.inRange(i + 1, j - 1) else array[i + 1][j - 1]

                    # Upper right: [0,0], [0, 1], [-1, 1], [-1, 0]
                    a1 = funH(x0, x1, x6, x2)
                    # Upper left: [0,0], [-1, 0], [-1, -1], [0, -1]
                    a2 = funH(x0, x2, x7, x3)
                    # Bottom left: [0,0], [0, -1], [1, -1], [1, 0]
                    a3 = funH(x0, x3, x8, x4)
                    # Bottom right: [0,0], [1, 0], [1, 1], [0, 1]
                    a4 = funH(x0, x4, x5, x1)
                    ret_array[i][j] = funF(a1, a2, a3, a4)
                else: ret_array[i][j] = 0
        return ret_array

    def pairRelationship(self, array):
        # Input: Yokoi (shape 64,64, integer)
        # Output: Pair Relationship result (shape 64,64, integer)
        def funH(a, m):
            if a == m:
                return 1
            else:
                return 0

        def funF(a1, a2, a3, a4):
            if (a1 + a2 + a3 + a4) < 1:
                return 'q'
            else:
                return 'p'

        ret_array = np.zeros((64,64)).astype(str)
        for i in range(64):
            for j in range(64):
                if array[i][j] == 1:
                    x0 = array[i][j]
                    x1 = 0 if not self.inRange(i + 0, j + 1) else array[i + 0][j + 1]
                    x2 = 0 if not self.inRange(i - 1, j + 0) else array[i - 1][j + 0]
                    x3 = 0 if not self.inRange(i + 0, j - 1) else array[i + 0][j - 1]
                    x4 = 0 if not self.inRange(i + 1, j + 0) else array[i + 1][j + 0]

                    a1 = funH(x0, x1)
                    a2 = funH(x0, x2)
                    a3 = funH(x0, x3)
                    a4 = funH(x0, x4)

                    ret_array[i][j] = funF(a1, a2, a3, a4)
                elif array[i][j] == 0:
                    ret_array[i][j] = '0'
                else:
                    ret_array[i][j] = 'q'
        return ret_array

    def connectedShrink(self, pr_array, binds_array):
        # Input: Pair Relationship(shape (64,64), str): '0':background, 'p': interesting, 'q':not interestin
        #        Binary Down-sample array(shape (64,64), int): 0:background, 1:foreground
        # Output: Connected Shrink result(shape (64,64), int): 0:background, 1:foreground
        def funH(b,c,d,e):
            if b == c and (d!=b or e!=b): return 1
            else: return 0
        def funF(a1,a2,a3,a4):
            if (a1+a2+a3+a4) == 1: # make it background
                self.cnt += 1
                return 0
            else:
                return 1
        ret_array = np.copy(binds_array)
        for i in range(64):
            for j in range(64):
                if pr_array[i][j] == 'p':
                    x0 = ret_array[i][j]
                    x1 = 0 if not self.inRange(i + 0, j + 1) else ret_array[i + 0][j + 1]
                    x2 = 0 if not self.inRange(i - 1, j + 0) else ret_array[i - 1][j + 0]
                    x3 = 0 if not self.inRange(i + 0, j - 1) else ret_array[i + 0][j - 1]
                    x4 = 0 if not self.inRange(i + 1, j + 0) else ret_array[i + 1][j + 0]
                    x5 = 0 if not self.inRange(i + 1, j + 1) else ret_array[i + 1][j + 1]
                    x6 = 0 if not self.inRange(i - 1, j + 1) else ret_array[i - 1][j + 1]
                    x7 = 0 if not self.inRange(i - 1, j - 1) else ret_array[i - 1][j - 1]
                    x8 = 0 if not self.inRange(i + 1, j - 1) else ret_array[i + 1][j - 1]

                    # Upper right: [0,0], [0, 1], [-1, 1], [-1, 0]
                    a1 = funH(x0, x1, x6, x2)
                    # Upper left: [0,0], [-1, 0], [-1, -1], [0, -1]
                    a2 = funH(x0, x2, x7, x3)
                    # Bottom left: [0,0], [0, -1], [1, -1], [1, 0]
                    a3 = funH(x0, x3, x8, x4)
                    # Bottom right: [0,0], [1, 0], [1, 1], [0, 1]
                    a4 = funH(x0, x4, x5, x1)
                    ret_array[i][j] = funF(a1, a2, a3, a4)
        return ret_array

if __name__ == '__main__':
    lena = Lena()