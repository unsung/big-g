import numpy as np
from data import Data

LEN = 640.*1167./1280.
C = LEN/1.325
D = 1.157*C
pi = np.pi
r = 0.05
d = 0.05
M = 1.438
L=4.235
S = 0
T0 = 0
    
def angle(x):
    return np.arctan(x/D)

class Big_G(object):
    def __init__(self):
        pass

    def run(self):
        return pi*pi*r*r*d*S/(M*T0*T0*L)



if __name__ == '__main__':
    import sys

    try:
        pathl = sys.argv[1]
        pathr = sys.argv[2]
    except: exit()

    _,Bl,Tl,_ = Data(pathl).run()
    _,Br,Tr,_ = Data(pathr).run()
    S = np.fabs(Bl-Br)/C
    T0 = 0.5*(Tl+Tr)
    print Big_G().run()
