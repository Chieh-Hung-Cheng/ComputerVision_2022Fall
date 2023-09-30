import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


class Lena():
    def __init__(self):
        self.pic = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
        self.Execute()

    def Execute(self):
        # roberts = self.Roberts(self.pic)
        robersKernels = [(((0,0), -1), ((0,1), 0),
                         ((1,0), 0), ((1,1), 1)),

                         (((0, 0), 0), ((0, 1), -1),
                         ((1, 0), 1), ((1, 1), 0))]
        roberts = self.DetectEdge(self.pic, robersKernels, 12, 'SQRT', ksize=2)
        self.ShowImg(roberts, save=True, filename='roberts12')

        # prewitt = self.Prewitt(self.pic)
        prewittKernels = [(((-1, -1), -1), ((-1, 0), -1), ((-1, 1), -1),
                           ((1, -1), 1), ((1, 0), 1), ((1, 1), 1)),

                          (((-1, -1), -1), ((-1, 1), 1),
                          ((0, -1), -1), ((0, 1), 1),
                          ((1, -1), -1), ((1, 1), 1),)]
        prewitt = self.DetectEdge(self.pic, prewittKernels, 24, 'SQRT')
        self.ShowImg(prewitt, save=True, filename='prewitt24')
        sobelKernels = [(((-1, -1), -1), ((-1, 0), -2), ((-1, 1), -1),
                           ((1, -1), 1), ((1, 0), 2), ((1, 1), 1)),

                          (((-1, -1), -1), ((-1, 1), 1),
                          ((0, -1), -2), ((0, 1), 2),
                          ((1, -1), -1), ((1, 1), 1),)]
        sobel = self.DetectEdge(self.pic, sobelKernels, 38, 'SQRT')
        self.ShowImg(sobel, save=True, filename='sobel38')

        freiChenKernels = [(((-1, -1), -1), ((-1, 0), -1*np.sqrt(2)), ((-1, 1), -1),
                         ((1, -1), 1), ((1, 0), np.sqrt(2)), ((1, 1), 1)),

                        (((-1, -1), -1), ((-1, 1), 1),
                         ((0, -1), -1*np.sqrt(2)), ((0, 1), np.sqrt(2)),
                         ((1, -1), -1), ((1, 1), 1),)]
        freichen = self.DetectEdge(self.pic, freiChenKernels, 30, 'SQRT')
        self.ShowImg(freichen, save=True, filename='freichen38')

        kirschKernels = [(((-1, -1), -3), ((-1, 0), -3), ((-1, 1), 5),
                          ((0, -1), -3),                  ((0, 1), 5),
                          ((1, -1), -3), ((1, 0), -3), ((1, 1), 5)),

                         (((-1, -1), -3), ((-1, 0), 5), ((-1, 1), 5),
                          ((0, -1), -3),                 ((0, 1), 5),
                          ((1, -1), -3), ((1, 0), -3), ((1, 1), -3)),

                         (((-1, -1), 5), ((-1, 0), 5), ((-1, 1), 5),
                          ((0, -1), -3),               ((0, 1), -3),
                          ((1, -1), -3), ((1, 0), -3), ((1, 1), -3)),

                         (((-1, -1), 5), ((-1, 0), 5), ((-1, 1), -3),
                          ((0, -1), 5),                 ((0, 1), -3),
                          ((1, -1), -3), ((1, 0), -3), ((1, 1), -3)),

                         (((-1, -1), 5), ((-1, 0), -3), ((-1, 1), -3),
                          ((0, -1), 5),                 ((0, 1), -3),
                          ((1, -1), 5), ((1, 0), -3), ((1, 1), -3)),

                         (((-1, -1), -3), ((-1, 0), -3), ((-1, 1), -3),
                          ((0, -1), 5),                 ((0, 1), -3),
                          ((1, -1), 5), ((1, 0), 5), ((1, 1), -3)),

                         (((-1, -1), -3), ((-1, 0), -3), ((-1, 1), -3),
                          ((0, -1), -3),                ((0, 1), -3),
                          ((1, -1), 5), ((1, 0), 5), ((1, 1), 5)),

                         (((-1, -1), -3), ((-1, 0), -3), ((-1, 1), -3),
                          ((0, -1), -3),                ((0, 1), 5),
                          ((1, -1), -3), ((1, 0), 5), ((1, 1), 5))
                           ]
        krisch = self.DetectEdge(self.pic, kirschKernels, 135, 'MAX')
        self.ShowImg(krisch, save=True, filename='krisch135')

        robinsonKernels = [(((-1, -1), -1),         ((-1, 1), 1),
                             ((0, -1), -2),         ((0, 1), 2),
                             ((1, -1), -1),         ((1, 1), 1)),

                         ((                ((-1, 0), 1), ((-1, 1), 2),
                          ((0, -1), -1),                 ((0, 1), 1),
                          ((1, -1), -2), ((1, 0), -1)              )),

                         (((-1, -1), 1), ((-1, 0), 2), ((-1, 1), 1),

                          ((1, -1), -1), ((1, 0), -2), ((1, 1), -1)),

                         (((-1, -1), 2), ((-1, 0), 1),
                          ((0, -1), 1),                ((0, 1), -1),
                                         ((1, 0), -1), ((1, 1), -2)),

                         (((-1, -1), 1),                ((-1, 1), -1),
                          ((0, -1), 2),                 ((0, 1), -2),
                          ((1, -1), 1),                 ((1, 1), -1)),

                         ((              ((-1, 0), -1), ((-1, 1), -2),
                          ((0, -1), 1),                 ((0, 1), -1),
                          ((1, -1), 2), ((1, 0), 1)                )),

                         (((-1, -1), -1), ((-1, 0), -2), ((-1, 1), -1),

                          ((1, -1), 1), ((1, 0), 2), ((1, 1), 1)),

                         (((-1, -1), -2), ((-1, 0), -1),
                          ((0, -1), -1),                 ((0, 1), 1),
                                         ((1, 0), 1), ((1, 1), 2))]
        robinson = self.DetectEdge(self.pic,robinsonKernels , 43, 'MAX')
        self.ShowImg(robinson, save=True, filename='robinson43')

        nevatiaBabuKernels = [(( (-2, -2), 100), ((-2, -1), 100), ((-2, 0), 100), ((-2, 1), 100),((-2, 2), 100),
                                ((-1, -2), 100), ((-1, -1), 100), ((-1, 0), 100), ((-1, 1), 100), ((-1, 2), 100),
                                ((0, -2), 0), ((0, -1), 0), ((0, 1), 0), ((0, 2), 0),
                                ((1, -2), -100), ((1, -1), -100), ((1, 0), -100), ((1, 1), -100), ((1, 2), -100),
                                ((2, -2), -100), ((2, -1), -100), ((2, 0), -100), ((2, 1), -100), ((2, 2), -100)),

                              (((-2, -2), 100), ((-2, -1), 100), ((-2, 0), 100), ((-2, 1), 100), ((-2, 2), 100),
                               ((-1, -2), 100), ((-1, -1), 100), ((-1, 0), 100), ((-1, 1), 78), ((-1, 2), -32),
                               ((0, -2), 100), ((0, -1), 92), ((0, 1), -92), ((0, 2), -100),
                               ((1, -2), 32), ((1, -1), -78), ((1, 0), -100), ((1, 1), -100), ((1, 2), -100),
                               ((2, -2), -100), ((2, -1), -100), ((2, 0), -100), ((2, 1), -100), ((2, 2), -100)),

                              (((-2, -2), 100), ((-2, -1), 100), ((-2, 0), 100), ((-2, 1), 32), ((-2, 2), -100),
                               ((-1, -2), 100), ((-1, -1), 100), ((-1, 0), 92), ((-1, 1), -78), ((-1, 2), -100),
                               ((0, -2), 100), ((0, -1), 100), ((0, 1), -100), ((0, 2), -100),
                               ((1, -2), 100), ((1, -1), 78), ((1, 0), -92), ((1, 1), -100), ((1, 2), -100),
                               ((2, -2), 100), ((2, -1), -32), ((2, 0), -100), ((2, 1), -100), ((2, 2), -100)),

                              (((-2, -2), -100), ((-2, -1), -100), ((-2, 0), 0), ((-2, 1), 100), ((-2, 2), 100),
                               ((-1, -2), -100), ((-1, -1), -100), ((-1, 0), 0), ((-1, 1), 100), ((-1, 2), 100),
                               ((0, -2), -100), ((0, -1), -100), ((0, 1), 100), ((0, 2), 100),
                               ((1, -2), -100), ((1, -1), -100), ((1, 0), 0), ((1, 1), 100), ((1, 2), 100),
                               ((2, -2), -100), ((2, -1), -100), ((2, 0), 0), ((2, 1), 100), ((2, 2), 100)),

                              (((-2, -2), -100), ((-2, -1), 32), ((-2, 0), 100), ((-2, 1), 100), ((-2, 2), 100),
                               ((-1, -2), -100), ((-1, -1), -78), ((-1, 0), 92), ((-1, 1), 100), ((-1, 2), 100),
                               ((0, -2), -100), ((0, -1), -100), ((0, 1), 100), ((0, 2), 100),
                               ((1, -2), -100), ((1, -1), -100), ((1, 0), -92), ((1, 1), 78), ((1, 2), 100),
                               ((2, -2), -100), ((2, -1), -100), ((2, 0), -100), ((2, 1), -32), ((2, 2), 100)),

                              (((-2, -2), 100), ((-2, -1), 100), ((-2, 0), 100), ((-2, 1), 100), ((-2, 2), 100),
                               ((-1, -2), -32), ((-1, -1), 78), ((-1, 0), 100), ((-1, 1), 100), ((-1, 2), 100),
                               ((0, -2), -100), ((0, -1), -92), ((0, 1), 92), ((0, 2), 100),
                               ((1, -2), -100), ((1, -1), -100), ((1, 0), -100), ((1, 1), -78), ((1, 2), 32),
                               ((2, -2), -100), ((2, -1), -100), ((2, 0), -100), ((2, 1), -100), ((2, 2), -100))]
        nevatiaBabu = self.DetectEdge(self.pic,nevatiaBabuKernels , 12500, 'MAX', ksize=5)
        self.ShowImg(nevatiaBabu, save=True, filename='navatiababu12500')
    def ShowImg(self, image, save=False, show=True, filename=None):
        '''if show:
            cv2.imshow('IMAGE', image.astype(np.uint8))
            cv2.waitKey()'''
        if save: cv2.imwrite('{}.bmp'.format(filename),image.astype(np.uint8))

    def PrintImg(self, array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j]!= 0 and array[i][j] != '0': print(array[i][j], end='')
                else: print('_', end='')
            print()
        print()

    def ISInRange(self, x, y):
        if x >= 512 or x < 0:
            return False
        if y >= 512 or y < 0:
            return False
        return True

    def Roberts(self, inPic, thr=30):
        '''
        :param inPic: input picture
        :param thr: threshold
        :return: output picture
        kernel:
        r1: [-1 0
             0 1]
        r2: [0 -1
             1 0]
        '''
        retPic = np.zeros_like(inPic)
        paddedPic = cv2.copyMakeBorder(inPic, 0, 1, 0, 1,cv2.BORDER_REPLICATE)
        for i in range(inPic.shape[0]):
            for j in range(inPic.shape[1]):
                r1 = -1*paddedPic[i][j] + 1*paddedPic[i+1][j+1]
                r2 = -1*paddedPic[i][j+1] + 1*paddedPic[i+1][j]
                grad = np.sqrt(np.power(r1, 2) + np.power(r2, 2))
                if grad>=thr: retPic[i][j]=0
                else: retPic[i][j]=255
        return retPic

    def Prewitt(self, inPic, thr=24):
        retPic = np.zeros_like(inPic)
        paddedPic = cv2.copyMakeBorder(inPic, 1, 1, 1, 1,cv2.BORDER_REPLICATE)
        p1Kernel = [((-1, -1), -1), ((-1, 0), -1), ((-1, 1), -1),
                  ((1, -1), 1), ((1, 0), 1), ((1, 1), 1)]
        p2Kernel = [((-1, -1), -1), ((-1, 1), 1),
                  ((0, -1), -1), ((0, 1), 1),
                  ((1, -1), -1), ((1, 1), 1),]
        for i in range(inPic.shape[0]):
            for j in range(inPic.shape[1]):
                p1 = self.ApplyKernel(paddedPic, p1Kernel, i+1, j+1)
                p2 = self.ApplyKernel(paddedPic, p2Kernel, i + 1, j + 1)
                grad = np.sqrt(np.power(p1, 2)+np.power(p2, 2))
                if grad>=thr: retPic[i][j] = 0
                else: retPic[i][j]=255
        return retPic

    def ApplyKernel(self, img, kernel, row, col):
        retVal = 0
        for elm in kernel:
            pos = elm[0]
            val = elm[1]
            retVal += val * img[row+pos[0]][col+pos[1]]
        return retVal

    def DetectEdge(self, inPic, kernelLst, thr, method, ksize=3):
        retPic = np.zeros_like(inPic)
        paddedPic = None
        if ksize==3: paddedPic = cv2.copyMakeBorder(inPic, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        elif ksize==5: paddedPic = cv2.copyMakeBorder(inPic, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
        elif ksize==2: paddedPic = cv2.copyMakeBorder(inPic, 0, 1, 0, 1, cv2.BORDER_REPLICATE)
        assert paddedPic is not None

        for i in range(inPic.shape[0]):
            for j in range(inPic.shape[1]):
                valLst = []
                for kernel in kernelLst:
                    val = 0
                    if ksize==3: val = self.ApplyKernel(paddedPic, kernel, i+1, j+1)
                    elif ksize==5: val = self.ApplyKernel(paddedPic, kernel, i+2, j+2)
                    elif ksize==2: val = self.ApplyKernel(paddedPic, kernel, i, j)
                    valLst.append(val)
                if method == 'SQRT':
                    grad = np.sqrt(np.sum(np.power(valLst, 2)))
                elif method == 'MAX':
                    grad = np.max(valLst)
                if grad>=thr: retPic[i][j]=0
                else: retPic[i][j]=255
        return retPic



if __name__ == '__main__':
    lena = Lena()