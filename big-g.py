import numpy as np
import cv2
from data import Data

const = 1.0107
LEN = 640.*1167./1280.
C = LEN/1.325
D = 1.157*C
pi = np.pi
r = 0.045
d = 0.05
M = 1.438
L = 4.235
S = 0
T0 = 0
    
def find_g():
    return pi*pi*r*r*d*S/(M*T0*T0*L)



if __name__ == '__main__':
    import sys

    try:
        pathl = sys.argv[1]
        pathr = sys.argv[2]
    except: exit()

    valsl, sdevl, xl,tl, valsl2, sdevl2 = Data(pathl).run()
    valsr, sdevr, xr,tr, valsr2, sdevr2 = Data(pathr).run()
    """
    valsl, sdevl, valsl2, sdevl2 = Data(pathl).run()
    valsr, sdevr, valsr2, sdevr2 = Data(pathr).run()
    """
    _,Bl, Tl,_ = valsl
    _,Br, Tr,_ = valsr
    _,Bl2, Tl2,_,bl2 = valsl2
    _,Br2, Tr2,_,br2 = valsr2
    _,sBl, sTl,_ = sdevl
    _,sBr, sTr,_ = sdevr
    _,sBl2, sTl2,_,sbl2 = sdevl2
    _,sBr2, sTr2,_,sbr2 = sdevr2
    S = np.fabs(Bl-Br)/C
    T0 = Tl
    tmax=np.amax([tl[-1],tr[-1]])
    xmin=np.amin([np.amin(xl), np.amin(xr)])
    xmax=np.amax([np.amax(xl), np.amax(xr)])
    plot = np.zeros((int(xmax-xmin), int(tmax), 3), np.uint8)
    print plot.shape
    for i in np.arange(xl.size-1):
        plot[np.clip(int(xl[i]-xmin),0,plot.shape[0]-1),int(tl[i])]=(0,0,255)
    for i in np.arange(xr.size-1):
        plot[np.clip(int(xr[i]-xmin),0,plot.shape[0]-1),int(tr[i])]=(0,255,0)

    cv2.namedWindow('wave')
    cv2.imshow('wave',plot)
    print find_g()
    while cv2.waitKey(0)&0xFF is not 27:
        pass
