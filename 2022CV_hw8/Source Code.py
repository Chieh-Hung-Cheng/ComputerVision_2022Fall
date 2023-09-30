import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


class Lena():
    def __init__(self):
        self.pic = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
        self.ShowImg(self.pic, show=False)
        self.GenNoisePic()
        self.noisePicDict = {'Gau10': cv2.imread('Gaussian_10.bmp', cv2.IMREAD_GRAYSCALE),
                             'Gau30': cv2.imread('Gaussian_30.bmp', cv2.IMREAD_GRAYSCALE),
                             'SP.05': cv2.imread('SaltPepper_0.1.bmp', cv2.IMREAD_GRAYSCALE),
                             'SP.1': cv2.imread('SaltPepper_0.1.bmp', cv2.IMREAD_GRAYSCALE),}
        self.kernel = [(-2, -1), (-2, 0), (-2, 1),
                (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
                (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
                (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
                        (2, -1), (2, 0), (2, 1)]
        self.Execute()

    def Execute(self):
        for name, img in self.noisePicDict.items():
            for fn in ['mean', 'median']:
                for sz in [3,5]:
                    resultPic = self.ApplyFilter(img, size=sz, fun=fn)
                    resultSNR = self.CalcSNR(self.pic, resultPic)
                    self.ShowImg(resultPic, show=False,
                                 save=True, filename='{}_{}_{}'.format(name, fn, sz))
                    print(name, fn, sz,' SNR: ', resultSNR)
        for name, img in self.noisePicDict.items():
            openClosePic = self.closing(self.opening(img))
            openCloseSNR = self.CalcSNR(self.pic, openClosePic)
            self.ShowImg(openClosePic, show=False, save=True,
                         filename='{}_OpenThenClose'.format(name))
            print("OpenThenClose SNR",name,' SNR: ', openCloseSNR)

            closeOpenPic = self.opening(self.closing(img))
            closeOpenSNR = self.CalcSNR(self.pic, closeOpenPic)
            self.ShowImg(closeOpenPic, show=False, save=True,
                         filename='{}_CloseThenOpen'.format(name))
            print("CloseThenOpen SNR",name,' SNR: ', closeOpenSNR)

    def CalcSNR(self, oriPic, noisePic):
        mu_ori = np.sum(oriPic.astype(float)) / oriPic.size
        vs = np.sum(np.power(oriPic.astype(float)-mu_ori, 2)) / oriPic.size
        mu_noise = np.sum(noisePic.astype(float)-oriPic.astype(float)) / oriPic.size
        vn = np.sum(np.power(noisePic.astype(float)-oriPic.astype(float)-mu_noise ,2)) / oriPic.size
        # vs = np.var(oriPic)
        # vn = np.var(noisePic-oriPic)
        snr = 20*np.log10(np.sqrt(vs)/np.sqrt(vn))
        return snr

    def ApplyFilter(self, inPic, size=3, fun='mean'):
        def GetFromKernel(img, i, j):
            valList = None
            if size == 3:
                valList = [img[i-1][j-1], img[i-1][j], img[i-1][j+1],
                           img[i][j-1], img[i][j], img[i][j+1],
                           img[i+1][j-1], img[i+1][j], img[i+1][j+1]]
            elif size == 5 :
                valList = [img[i-2][j-2],img[i-2][j-1], img[i-2][j], img[i-2][j+1],img[i-2][j+2],
                           img[i-1][j-2],img[i-1][j-1], img[i-1][j], img[i-1][j+1],img[i-1][j+2],
                           img[i][j-2],img[i][j-1], img[i][j], img[i][j+1],img[i][j+2],
                           img[i+1][j-2],img[i+1][j-1], img[i+1][j], img[i+1][j+1],img[i+1][j+2],
                           img[i+2][j-2], img[i+2][j-1], img[i+2][j], img[i+2][j+1], img[i+2][j+2]]

            if fun == 'mean': return int(np.mean(valList))
            elif fun == 'median': return int(np.median(valList))

        ret_pic = np.zeros_like(inPic)
        if size == 3:
            paddedPic = cv2.copyMakeBorder(inPic,1,1,1,1,cv2.BORDER_REPLICATE)
            for i in range(512):
                for j in range(512):
                    ret_pic[i][j] = GetFromKernel(paddedPic, i+1, j+1)
            return ret_pic
        elif size == 5:
            paddedPic = cv2.copyMakeBorder(inPic,2,2,2,2,cv2.BORDER_REPLICATE)
            for i in range(512):
                for j in range(512):
                    ret_pic[i][j] = GetFromKernel(paddedPic, i+2, j+2)
            return ret_pic

    def GenNoisePic(self):
        ampList = [10,30]
        for amp in ampList:
            gaussianPic = self.GenAddGaussian(amp)
            gaussianSNR = self.CalcSNR(self.pic, gaussianPic)
            print('Gaussain ', amp, ' SNR: ', gaussianSNR)
            self.ShowImg(gaussianPic, show=False, save=True, filename='Gaussian_{}'.format(amp))

        thrList = [0.05, 0.1]
        for thr in thrList:
            saltPepperPic = self.GenSaltPepper(thr)
            saltPepperSNR = self.CalcSNR(self.pic, saltPepperPic)
            print('SaltPepper ', thr, ' SNR: ', saltPepperSNR)
            self.ShowImg(saltPepperPic, show=False, save=True, filename='SaltPepper_{}'.format(thr))

    def GenAddGaussian(self, amplitude):
        ret_pic = np.zeros_like(self.pic)
        for i in range(self.pic.shape[0]):
            for j in range(self.pic.shape[1]):
                newVal = self.pic[i][j] + amplitude*random.gauss(0,1)
                if newVal>255: ret_pic[i][j] = 255
                elif newVal<0: ret_pic[i][j] = 0
                else: ret_pic[i][j] = newVal
        return ret_pic

    def GenSaltPepper(self, threshold):
        ret_pic = np.zeros_like(self.pic)
        for i in range(self.pic.shape[0]):
            for j in range(self.pic.shape[1]):
                randNum = random.uniform(0,1)
                if randNum < threshold: ret_pic[i][j] = 0
                elif randNum > (1 - threshold): ret_pic[i][j] = 255
                else: ret_pic[i][j] = self.pic[i][j]
        return ret_pic


    def ShowImg(self, image, save=False, show=True, filename=None):
        if show:
            cv2.imshow('IMAGE', image.astype(np.uint8))
            cv2.waitKey()
        if save: cv2.imwrite('{}.bmp'.format(filename),image.astype(np.uint8))

    def PrintImg(self, array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j]!= 0 and array[i][j] != '0': print(array[i][j], end='')
                else: print('_', end='')
            print()
        print()

    def ISInRange(self, x, y):
        if x >= 512 or x<0:
            return False
        if y >= 512 or y<0:
            return False
        return True

    def getValInKernel(self, refer, row, col):
        intensity_lst = []
        for point in self.kernel:
            if self.ISInRange(row+point[0], col+point[1]):
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


if __name__ == '__main__':
    lena = Lena()