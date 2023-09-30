import cv2
import numpy as np
import matplotlib.pyplot as plt



class Picture():
    def __init__(self):
        self.pic = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
        self.kernel = [
                      (-2, -1), (-2, 0), (-2, 1),
            (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
            ( 0, -2), ( 0, -1), ( 0, 0), ( 0, 1), ( 0, 2),
            ( 1, -2), ( 1, -1), ( 1, 0), ( 1, 1), ( 1, 2),
                      ( 2, -1), ( 2, 0), ( 2, 1)
        ]
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

    def getValInKernel(self, refer, row, col):
        intensity_lst = []
        for point in self.kernel:
            if self.inRange(row+point[0], col+point[1]):
                intensity_lst.append(refer[row+point[0]][col+point[1]])
        return intensity_lst

    def dilation(self, tgtpic):
        dilation_pic = np.copy(tgtpic)
        for i in range(dilation_pic.shape[0]):
            for j in range(dilation_pic.shape[1]):
                dilation_pic[i][j] = max(self.getValInKernel(tgtpic, i, j))
        return dilation_pic

    def erosion(self, tgtpic):
        erosion_pic = np.copy(tgtpic)
        for i in range(erosion_pic.shape[0]):
            for j in range(erosion_pic.shape[1]):
                erosion_pic[i][j] = min(self.getValInKernel(tgtpic, i, j))
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


    def sequential(self):
        # Dilation
        dilation_pic = self.dilation(self.pic)
        cv2.imwrite('lena_dilation.bmp', dilation_pic)

        # Erosion
        erosion_pic = self.erosion(self.pic)
        cv2.imwrite('lena_erosion.bmp', erosion_pic)

        # Closing
        closing_pic = self.closing(self.pic)
        cv2.imwrite('lena_closing.bmp', closing_pic)

        # Opening
        opening_pic = self.opening(self.pic)
        cv2.imwrite('lena_opening.bmp', opening_pic)









if __name__ == '__main__':
    pic = Picture()