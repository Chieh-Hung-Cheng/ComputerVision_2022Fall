import cv2
import numpy as np
import matplotlib.pyplot as plt

def showImg(image):
    cv2.imshow('IMAGE', image)
    cv2.waitKey()



class Picture():
    def __init__(self):
        self.pic = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
        self.sequential()

    def sequential(self):
        # Q1
        showImg(self.pic)
        self.generateHistogram(self.pic, 'original', 'Histogram of the Original Image')

        # Q2
        pic_div3 = self.divideByThree()
        showImg(pic_div3)
        cv2.imwrite('lena_div3.bmp', pic_div3)
        histo_array = self.generateHistogram(pic_div3, 'divide3', 'Histogram of the Image with 1/3 Pixel Intensity', ret=True)

        # Q3
        pic_eq = self.equalization(pic_div3, histo_array)
        showImg(pic_eq)
        self.generateHistogram(pic_eq, 'equalized', 'Histogram of the Equalized Image')
        cv2.imwrite('lena_equalized.bmp', pic_eq)

    def generateHistogram(self, img, filename, graphname, ret=False):
        def plotHistogram(ys, title='', save=True):
            plt.bar(range(256), ys)
            plt.title(graphname)
            plt.xlabel('Intensity')
            plt.ylabel('Number of Pixels')
            plt.savefig('histogram_{}.png'.format(filename))
            plt.clf()

        count_array = np.zeros((256,), dtype=np.uint32)
        for i in range(512):
            for j in range(512):
                count_array[img[i][j]] += 1
        plotHistogram(count_array, title=filename)
        if ret:
            return count_array


    def divideByThree(self):
        pic_new = np.zeros(self.pic.shape, np.uint8)
        for i in range(512):
            for j in range(512):
                pic_new[i][j] = int(self.pic[i][j]/3)
        return pic_new

    def equalization(self, pic_div3, histo_array):
        s_array = np.zeros((256,), dtype=float)

        cumulate = 0
        for k in range(256):
            cumulate += 255 * histo_array[k] / (512**2)
            s_array[k] = cumulate

        pic_new = np.zeros(self.pic.shape, np.uint8)
        for i in range(512):
            for j in range(512):
                pic_new[i][j] = int(s_array[pic_div3[i][j]])
        return pic_new




if __name__ == '__main__':
    pic = Picture()