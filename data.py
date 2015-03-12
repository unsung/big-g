import numpy as np
import cv2
import os
from scipy.optimize import curve_fit
from scipy.signal import gaussian

LEN = 640.*1167./1280.
C = LEN/1.325
D = 1.157*C
pi = np.pi
r = 0.045
d = 0.05
M = 1.438
L = 4.235
m = 0.015
I = 2*m*d*d
dy = 5

"""
LEN = 1235
C = LEN/1.322
D = 1.221*C
pi = np.pi
r = 0.045
d = 0.05
M = 1.438
L = 4.235
m = 0.015
I = 2*m*d*d
dy = 10
"""

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
        self.showimg = None
        self.calcimg = None
        self.data = None
        self.starttime = None
        self.currtime = None

    def hrange(self):
        return np.arange(self.h-dy, self.h+dy+1)

    def gendata(self,start=320,again=True):
        self.w = self.img.shape[1]
        self.y = self.img[self.hrange(),:,2].sum(axis=0)/(2.*dy)
        plot = np.zeros((256, self.w, 3), np.uint8)
        for i in range(self.w):
            plot[255-self.y[i], i]=(0,0,255)
        self.calcimg = np.copy(plot)
        if self.analyzing:
            params = None
            try:
                params = self.fit(start)[0]
                params = np.fabs(params)
                if self.data is not None and np.fabs((params[0]-LEN/2-self.data[-1][1])/(self.currtime-self.data[-1][0])*5) > 100:
                    params = None;

                if params[1] > 20:
                    params = None

                if params[2] / (params[1] * np.sqrt(2 * np.pi)) < 30:
                    params = None

            except: return

            if params is not None:
                #print params
                new = [[
                        self.currtime,
                        params[0]-LEN/2,
                        params[1]
                    ]]
                if self.data is not None:
                    self.data = np.append(self.data, new, axis=0)
                else:
                    self.data = np.array(new)
                #for x in np.arange(1280):
                for x in np.arange(640):
                    self.calcimg[255-self.gauss(x,params[0],params[1],params[2]),x]=(0,255,0)
                cv2.imshow('data', self.calcimg)
            else:
                if(again):
                    self.gendata(0,False)
                    self.gendata(self.w,False)
                    cv2.imshow('data', self.calcimg)
                else:
                    return

    def dispimg(self):
        self.showimg = np.copy(self.img)
        """
        cv2.line(self.showimg, (0, self.h), (1279, self.h), (0,0,255))
        cv2.line(self.showimg, (0, self.h-dy), (1279, self.h-dy), (0,255,0))
        cv2.line(self.showimg, (0, self.h+dy), (1279, self.h+dy), (0,255,0))
        """
        cv2.line(self.showimg, (0, self.h),    (639, self.h), (0,0,255))
        cv2.line(self.showimg, (0, self.h-dy), (639, self.h-dy), (0,255,0))
        cv2.line(self.showimg, (0, self.h+dy), (639, self.h+dy), (0,255,0))

        cv2.imshow('pic', self.showimg)

    def onmousepic(self, event, x, y, flags, param):
        if self.analyzing:
            return

        x, y = np.int16([x, y])

        if event == cv2.EVENT_LBUTTONDOWN:
            self.h = y
            self.gendata()
            self.dispimg()

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.analyzing = not (self.analyzing or self.done)
            print "Analyzing: ", self.analyzing

    def onmousedata(self, event, x, y, flags, param):
        pass

    #def gauss(self, x, u, s, A, off):
    def gauss(self, x, u, s, A):
        return np.mean(self.y) + A / (s * np.sqrt(2 * np.pi)) *\
            np.exp(-0.5 * np.power((x - u) / s, 2.))

    def fit(self,start):
        return curve_fit(
                self.gauss,
                range(self.w),
                self.y,
                p0=[start,10,1000]
                )

    def sin(self, t, A, B, T, d):
        return A*np.sin(2.*np.pi*t/T + d) + B

    def calcsin(self):
        t,x,s= self.data.T
        return curve_fit(self.sin, t, x, p0 = [300,1,300,1], sigma = s, absolute_sigma=True)
        #return curve_fit(self.sin, t, x, p0 = [500,1,600,1], sigma = s, absolute_sigma=True)

    def dampsin(self, t, A, B, T, d, b):
        return A*np.exp(-b*t/(2*I))*np.sin(2.*np.pi*t/T + d) + B

    def calcdampsin(self):
        t,x,s= self.data.T
        return curve_fit(self.dampsin, t, x, p0 = [300,1,300,1,.11], sigma = s, absolute_sigma=True)
        #return curve_fit(self.dampsin, t, x, p0 = [500,1,600,1,.11], sigma = s, absolute_sigma=True)

    def run(self):
        for imgname in sorted(os.listdir(self.path)):

            self.currtime = float(imgname[:-4])
            if self.starttime is None:
                self.starttime = self.currtime
            self.currtime -= self.starttime

            imgname = self.path+imgname
            self.img = cv2.imread(imgname)

            if self.showimg is None:
                cv2.imshow('pic', self.img)
            else:
                self.dispimg()

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

        t,x,s = self.data.T
        cv2.destroyAllWindows()
        vals,cov = self.calcsin()
        vals2,cov2 = self.calcdampsin()
        sdev = np.sqrt(np.clip(np.diag(cov),0,9999))
        sdev2 = np.sqrt(np.clip(np.diag(cov2),0,9999))
        return (vals,sdev,x,t,vals2,sdev2)

if __name__ == '__main__':
    import sys

    try: path = sys.argv[1]
    except: exit()

    print Data(path).run()


