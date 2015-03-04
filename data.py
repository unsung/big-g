import numpy as np
import cv2
import os
from scipy.optimize import curve_fit
from scipy.signal import gaussian

LEN = 640
C = LEN/1.325
D = 1.157*C
    
def angle(x):
    return np.arctan(x/D)


class Data(object):
    def __init__(self, path):
        self.path = './'+path+'/'
        cv2.namedWindow('pic')
        cv2.setMouseCallback('pic', self.onmousepic)
        cv2.namedWindow('data')
        cv2.setMouseCallback('data', self.onmousedata)
        self.w = None
        self.h = None
        self.y = None
        self.mode = 'R'
        self.analyzing = False
        self.running = False
        self.done = False
        self.img = None
        self.data = None
        self.starttime = None;
        self.currtime = None;

    def gendata(self):
        self.w = self.img.shape[1]
        self.y = self.img[self.h,:,2]
        plot = np.zeros((256, self.w, 3), np.uint8)
        for i in range(self.w):
            plot[255-self.y[i], i]=(0,0,255)
        cv2.imshow('data', plot)
        if self.analyzing:
            params = None
            try:
                params = self.fit()[0]
                params = np.fabs(params)
                if params[1] > 10: params = None
            except: return

            if params is not None:
                new = [[
                        self.currtime,
                        params[0]-LEN/2,
                        params[1]
                    ]]
                if self.data is not None:
                    self.data = np.append(self.data, new, axis=0)
                else:
                    self.data = np.array(new)

    def onmousepic(self, event, x, y, flags, param):
        if self.analyzing:
            return

        x, y = np.int16([x, y])

        if event == cv2.EVENT_LBUTTONDOWN:
            self.h = y
            self.gendata()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.analyzing = not (self.analyzing or self.done)
            print "Analyzing: ", self.analyzing

    def onmousedata(self, event, x, y, flags, param):
        pass

    def gauss(self, x, u, s, A, off):
        return off + A * np.power(s * np.sqrt(2 * np.pi), 2.) *\
            np.exp(-0.5 * np.power((x - u) / s, 2.))

    def fit(self):
        return curve_fit(
                self.gauss,
                range(self.w),
                self.y,
                p0=[self.w/2,50,50,50]
                )

    def sin(self, t, A, B, T, d):
        return A*np.sin(2.*np.pi*t/T + d) + B

    def calcsin(self):
        t,x,s= self.data.T
        return curve_fit(self.sin, t, x, p0 = [1,0,600,0], sigma = s, absolute_sigma=True)

    def run(self):
        for imgname in sorted(os.listdir(self.path)):

            self.currtime = float(imgname[:-4])
            if self.starttime is None:
                self.starttime = self.currtime
            self.currtime -= self.starttime

            imgname = self.path+imgname
            self.img = cv2.imread(imgname)
            cv2.imshow('pic', self.img)

            if self.h is not None:
                self.gendata()

            cv2.waitKey(1)

            while not self.running:
                k = cv2.waitKey(0)&0xFF
                if k == 27:
                    return
                if k == ord('n'):
                    self.running = self.analyzing
                    break

        while cv2.waitKey(0)&0xFF is not 27:
            pass

        cv2.destroyAllWindows()
        return self.calcsin()[0]

if __name__ == '__main__':
    import sys

    try: path = sys.argv[1]
    except: exit()

    print Data(path).run()


