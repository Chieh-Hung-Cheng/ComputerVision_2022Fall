import cv2
import numpy as np
import matplotlib.pyplot as plt

def showImg(image):
    cv2.imshow('IMAGE', image)
    cv2.waitKey()


def visByDigit(array, height=512, length=512):
    print('xxx     ', end='')
    for x in range(length):
        print('%06d' % x, end=' ')
    print('')
    for i in range(height):
        print('row %03d' % i, end=' ')
        for j in range(length):
            print('%06d' % array[i][j], end=' ')
        print('')


class Pictrue():
    def __init__(self):
        self.pic = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
        # showImg(self.pic)

    def binarize(self, save=True):
        pic_new = np.zeros_like(self.pic)
        rownum = self.pic.shape[0]
        colnum = self.pic.shape[1]
        for i in range(rownum):
            for j in range(colnum):
                pic_new[i][j] = (self.pic[i][j] > 128)
        # self.visByDigit(pic_new)
        pic_out = pic_new * 255
        showImg(pic_out)
        visByDigit(pic_new)
        if save:
            cv2.imwrite('lena_binarize.bmp', pic_out)
            np.save('binary_array.npy', pic_new)


    def histoize(self):
        plt.hist(self.pic.flatten(), bins='auto')
        plt.title("Histogram of lena")
        plt.savefig('lena_histogram.png')
        plt.show()

    def histoize_hardcore(self):
        count_array = np.zeros((255,), dtype=np.uint32)
        for i in range(512):
            for j in range(512):
                count_array[self.pic[i][j]] += 1
        np.savetxt("histogram.csv", count_array, delimiter=",")


class UF():
    def __init__(self):
        # Picture
        self.pic = cv2.imread('lena.bmp')
        # Initialize the cc_array(Connected Component Array), making each pixel a Connected Component itself
        self.binary_array = np.load('binary_array.npy')

        self.cc_array = np.zeros_like(self.binary_array, np.uint32)
        cc_ini = 1  # Initialization
        for i in range(512):
            for j in range(512):
                if self.binary_array[i][j] == 1:
                    self.cc_array[i][j] = cc_ini
                    cc_ini += 1
                else:
                    self.cc_array[i][j] = 0

        self.idx_max = np.sum(np.sum(self.binary_array, axis=1), axis=0)
        self.tree = np.zeros((self.idx_max, ), dtype=np.uint32)
        for idx, val in enumerate(self.tree):
            self.tree[idx] = idx
        self.counts = np.zeros((self.idx_max, ))
        self.bbox_idxes = None

    def findRoot(self, x):
        while(x != self.tree[x]):
            x = self.tree[x]
        return x

    def connected(self, x, y):
        if self.findRoot(x) == self.findRoot(y):
            return True
        else:
            return False

    def union(self, pair):
        if not self.connected(pair[0], pair[1]):
            self.tree[self.findRoot(pair[1])] = self.findRoot(pair[0])


    def propagate(self):
        # Propagate the cc_array
        trans = []
        for i in range(512):
            for j in range(512):
                if self.cc_array[i][j] != 0:
                    above = 0
                    left = 0
                    # Check above
                    if i != 0:
                        if self.cc_array[i-1][j] != 0:
                            above = self.cc_array[i-1][j]
                    # Check left
                    if j != 0:
                        if self.cc_array[i][j-1] != 0:
                            left = self.cc_array[i][j-1]

                    if above != 0 and left != 0:
                        self.cc_array[i][j] = min(above, left)

                        if above != left:
                            hi = max(above, left)
                            lo = min(above, left)
                            if tuple((lo, hi)) not in trans:
                                trans.append((lo, hi))
                    elif above != 0:
                        self.cc_array[i][j] = above
                    elif left != 0:
                        self.cc_array[i][j] = left
                    else:
                        pass

        for pair in trans:
            self.union(pair)


    def translate(self):
        for i in range(512):
            for j in range(512):
                if self.cc_array[i][j] != self.findRoot(self.cc_array[i][j]):
                    self.cc_array[i][j] = self.findRoot(self.cc_array[i][j])

    def countAndFilter(self):
        for i in range(512):
            for j in range(512):
                if self.cc_array[i][j] !=0:
                    self.counts[self.cc_array[i][j]] += 1
        self.bbox_idxes = np.where(self.counts > 500)[0]

    def drawBbox(self):
        print(self.bbox_idxes)
        color_lst = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        for ii, idx in enumerate(self.bbox_idxes):
            rw = []
            cl = []
            for i in range(512):
                for j in range(512):
                    if self.cc_array[i][j] == idx:
                        rw.append(i)
                        cl.append(j)
            cv2.rectangle(self.pic, (min(cl), min(rw)), (max(cl), max(rw)), color_lst[ii], 4)
            cv2.circle(self.pic, (int(np.mean(cl)), int(np.mean(rw))), 5, color_lst[ii], 4)
        showImg(self.pic)
        cv2.imwrite('lena_output.bmp', self.pic)





if __name__ == '__main__':
    pic = Pictrue()
    pic.histoize_hardcore()
