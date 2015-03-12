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
m = 0.015
I = 2*m*d*d

"""
const = 1.0107
LEN = 1235
C = LEN/1.322
D = 1.221*C
pi = np.pi
r = 0.045
d = 0.05
M = 1.438
L = 4.235
S = 0
T0 = 0
m = 0.015
I = 2*m*d*d
"""

def find_g():
    return pi*pi*r*r*d*S/(M*T0*T0*L)

def find_dev():
    return pi*pi*r*r*d/(M*L*T0*T0)*np.sqrt(np.square(sBl)+np.square(sBr)+4*S/(T0*T0)*(np.square(sTl)+np.square(sTl)))

def dampsin(t, A, B, T, d, b):
    return A*np.exp(-b*t/(2*I))*np.sin(2.*np.pi*t/T + d) + B

def sin(t, A, B, T, d):
    return A*np.sin(2.*np.pi*t/T + d) + B



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
    Al,Bl, Tl,dl = valsl
    Ar,Br, Tr,dr = valsr
    Al2,Bl2, Tl2,dl2,bl2 = valsl2
    Ar2,Br2, Tr2,dr2,br2 = valsr2
    sAl,sBl, sTl,_ = sdevl
    sAr,sBr, sTr,_ = sdevr
    _,sBl2, sTl2,_,sbl2 = sdevl2
    _,sBr2, sTr2,_,sbr2 = sdevr2
    
    """
    T1 = np.average([Tl2,Tr2],weights=[np.power(sTl2,-2),np.power(sTr2,-2)])
    b = np.average([bl2,br2],weights=[np.power(sbl2,-2),np.power(sbr2,-2)])

    T0 = np.power(np.power(T1,-2)+np.power(b/(4*pi*I),2),-0.5)
    """
    T0 = np.average([Tl,Tr],weights=[sTl,sTr])
    """
    print b/(4*pi*I)
    print T1
    print T0
    """
    """
    print valsl
    print valsr

    """
    S = np.fabs(Bl-Br)/C
    tmax=np.amax([tl[-1],tr[-1]])
    xmin=np.amin([np.amin(xl), np.amin(xr)])
    xmax=np.amax([np.amax(xl), np.amax(xr)])
    #print xmin
    #print xmax
    """
    plot = np.zeros((int(xmax-xmin), int(tmax), 3), np.uint8)

    #print plot.shape
    for i in np.arange(xl.size-1):
        plot[np.clip(int(xl[i]-xmin),0,plot.shape[0]-1),int(tl[i])]=(0,0,255)
    for i in np.arange(xr.size-1):
        plot[np.clip(int(xr[i]-xmin),0,plot.shape[0]-1),int(tr[i])]=(0,255,0)
    """

    plot = np.zeros((1000, 16*60, 3), np.uint8)

    for i in np.arange(xl.size-1):
        plot[np.clip(int(xl[i]+500),0,plot.shape[0]-1),int(tl[i])]=(0,0,255)
    for i in np.arange(xr.size-1):
        plot[np.clip(int(xr[i]+500),0,plot.shape[0]-1),int(tr[i])]=(0,255,0)
        """
    for i in np.arange(plot.shape[1]-1):
        plot[int(dampsin(i,Al,Bl,Tl,dl,bl2))+500,i]=(255,0,0)
        plot[int(dampsin(i,Ar,Br,Tr,dr,br2))+500,i]=(255,0,0)
        """
    for i in np.arange(plot.shape[1]-1):
        plot[int(sin(i,Al,Bl,Tl,dl))+500,i]=(255,0,0)
        plot[int(sin(i,Ar,Br,Tr,dr))+500,i]=(255,0,0)

    cv2.namedWindow('wave')
    cv2.imshow('wave',plot)
    print "Bl="+str(Bl)+"+-"+str(sBl)
    print "Br="+str(Br)+"+-"+str(sBr)
    print "S="+ str(S) +"+-"+str(np.sqrt(np.square(sBl)+np.square(sBr)))
    print "Tl="+str(Tl)+"+-"+str(sTl)
    print "Tr="+str(Tr)+"+-"+str(sTr)
    print "T0="+str(T0)+"+-"+str(np.sqrt(np.square(sTl)+np.square(sTl)))
    print "Al="+str(Al/C)+"+-"+str(sAl/C)
    print "Ar="+str(Ar/C)+"+-"+str(sAr/C)

    print "G="+str(find_g()) + "+-" + str(find_dev())
    while cv2.waitKey(0)&0xFF is not 27:
        pass
