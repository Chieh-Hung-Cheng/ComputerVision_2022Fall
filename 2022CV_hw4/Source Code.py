import cv2
import numpy as np
import matplotlib.pyplot as plt





class Picture():
    def __init__(self):
        self.pic = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
        self.binary = (self.pic >= 128).astype(np.uint8)
        self.kernel = [
                      (-2, -1), (-2, 0), (-2, 1),
            (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
            ( 0, -2), ( 0, -1), ( 0, 0), ( 0, 1), ( 0, 2),
            ( 1, -2), ( 1, -1), ( 1, 0), ( 1, 1), ( 1, 2),
                      ( 2, -1), ( 2, 0), ( 2, 1)
        ]
        self.kernel_hit = [(0, -1), (0, 0),
                                    (1, 0)]
        self.kernel_miss = [(-1, 0), (-1, 1),
                                     ( 0, 1)]
        self.showImg(self.binary*255)
        cv2.imwrite('lena_binary.bmp', self.binary * 255)
        self.sequential()

    def showImg(self, image):
        cv2.imshow('IMAGE', image)
        cv2.waitKey()

    def inRange(self, x, y):
        if x >= self.pic.shape[0] or x<0:
            return False
        if y >= self.pic.shape[1] or y<0:
            return False
        return True

    def spread(self, array, row, col):
        for point in self.kernel:
            if self.inRange(row+point[0], col+point[1]):
                array[row+point[0]][col+point[1]] = 1


    def dilation(self, tgtpic):
        dilation_pic = np.copy(tgtpic)
        for i in range(dilation_pic.shape[0]):
            for j in range(dilation_pic.shape[1]):
                if tgtpic[i][j] == True:
                    self.spread(dilation_pic, i, j)
        return dilation_pic

    def checkFit(self, array, refer,  row, col):
        ret_bool = True
        for point in self.kernel:
            if self.inRange(row+point[0], col+point[1]):
                if not refer[row+point[0]][col+point[1]] == True:
                    ret_bool = False
                    break
        array[row][col] = ret_bool

    def erosion(self, tgtpic):
        erosion_pic = np.copy(tgtpic)
        for i in range(erosion_pic.shape[0]):
            for j in range(erosion_pic.shape[1]):
                if tgtpic[i][j] == True:
                    self.checkFit(erosion_pic, tgtpic, i, j)
        return erosion_pic

    def closing(self, tgtpic):
        # Dilation -> Erosion
        closing_pic = self.dilation(tgtpic)
        closing_pic = self.erosion(closing_pic)
        return closing_pic

    def opening(self, tgtpic):
        opening_pic = self.erosion(tgtpic)
        opening_pic = self.dilation(opening_pic)
        return opening_pic

    def match(self, row, col):
        ret_bool = True
        for point in self.kernel_hit:
            if self.inRange(row + point[0], col + point[1]):
                if not self.binary[row + point[0]][col + point[1]] == True:
                    ret_bool = False
                    break
        for point in self.kernel_miss:
            if self.inRange(row + point[0], col + point[1]):
                if self.binary[row + point[0]][col + point[1]] == True:
                    ret_bool = False
                    break
        return ret_bool

    def hit_miss(self):
        hitmiss_pic = np.copy(self.binary)
        for i in range(self.binary.shape[0]):
            for j in range(self.binary.shape[1]):
                if self.binary[i][j] == True:
                    hitmiss_pic[i][j] = self.match(i, j)
        return hitmiss_pic

    def sequential(self):
        # Dilation
        dilation_pic = self.dilation(self.binary)
        cv2.imwrite('lena_dilation.bmp', dilation_pic * 255)

        # Erosion
        erosion_pic = self.erosion(self.binary)
        cv2.imwrite('lena_erosion.bmp', erosion_pic * 255)

        # Closing
        closing_pic = self.closing(self.binary)
        cv2.imwrite('lena_closing.bmp', closing_pic * 255)

        # Opening
        opening_pic = self.opening(self.binary)
        cv2.imwrite('lena_opening.bmp', opening_pic * 255)

        # Hit & Miss
        hitmiss_pic = self.hit_miss()
        cv2.imwrite('lena_hitmiss.bmp', hitmiss_pic * 255)









if __name__ == '__main__':
    pic = Picture()