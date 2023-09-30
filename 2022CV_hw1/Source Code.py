import cv2
import imutils
import numpy as np

def showImg(image):
    cv2.imshow('IMAGE', image)
    cv2.waitKey()


class Pictrue():
    def __init__(self):
        self.pic = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
        showImg(self.pic)

    def upsideDown(self, save=True):
        pic_new = np.zeros_like(self.pic)
        rownum = self.pic.shape[0]
        colnum = self.pic.shape[1]
        for i in range(rownum-1):
            for j in range(colnum-1):
                pic_new[i][j] = self.pic[rownum-1-i][j]
        showImg(pic_new)
        if save: cv2.imwrite('lena_upsideDown.bmp', pic_new)

    def rightsideLeft(self, save=True):
        pic_new = np.zeros_like(self.pic)
        rownum = self.pic.shape[0]
        colnum = self.pic.shape[1]
        for i in range(rownum-1):
            for j in range(colnum-1):
                pic_new[i][j] = self.pic[i][colnum-1-j]
        showImg(pic_new)
        if save: cv2.imwrite('lena_rightsideLeft.bmp', pic_new)

    def diagonalFilp(self, save=True):
        pic_new = np.zeros_like(self.pic)
        rownum = self.pic.shape[0]
        colnum = self.pic.shape[1]
        for i in range(rownum-1):
            for j in range(colnum-1):
                pic_new[i][j] = self.pic[rownum-1-i][colnum-1-j]
        showImg(pic_new)
        if save: cv2.imwrite('lena_diagonalFlip.bmp', pic_new)

    def rotate45(self, save=True):
        pic_new = imutils.rotate(self.pic, -45)
        showImg(pic_new)
        if save: cv2.imwrite('lena_rotate45.bmp', pic_new)

    def shrinkHalf(self, save=True):
        h_new = int(self.pic.shape[0]/2)
        w_new = int(self.pic.shape[1]/2)
        pic_new = cv2.resize(self.pic, (h_new, w_new), cv2.INTER_AREA)
        showImg(pic_new)
        if save: cv2.imwrite('lena_shrinkHalf.bmp', pic_new)
        print('Shape for the new picture: ', pic_new.shape)

    def binarize(self, save=True):
        pic_new = np.zeros_like(self.pic)
        rownum = self.pic.shape[0]
        colnum = self.pic.shape[1]
        for i in range(rownum-1):
            pic_new[i] = (self.pic[i] > 128)*255
        showImg(pic_new)
        if save: cv2.imwrite('lena_binarize.bmp', pic_new)

if __name__ == '__main__':
    pic = Pictrue()
    # pic.upsideDown()
    # pic.rightsideLeft()
    # pic.diagonalFilp()
    # pic.rotate45()
    pic.shrinkHalf()
    # pic.binarize()
    x = 0