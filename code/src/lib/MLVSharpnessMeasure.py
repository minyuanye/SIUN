import numpy as np
from scipy.special import gamma
from skimage import color

class MLVMeasurement():
    def __init__(self):
        self.gam = np.linspace(0.2,10,9801)

    def __estimateggdparam(self,vec):
        gam = self.gam
        r_gam = (gamma(1/gam)*gamma(3/gam))/((gamma(2/gam)) ** 2)
        sigma_sq = np.mean(vec ** 2)
        sigma = np.sqrt(sigma_sq)
        return sigma

    def __MLVMap(self,img):
        xs, ys = img.shape
        x=img
        x1=np.zeros((xs,ys))
        x2=np.zeros((xs,ys))
        x3=np.zeros((xs,ys))
        x4=np.zeros((xs,ys))
        x5=np.zeros((xs,ys))
        x6=np.zeros((xs,ys))
        x7=np.zeros((xs,ys))
        x8=np.zeros((xs,ys))
        x9=np.zeros((xs,ys))
        x1[0:xs-2,0:ys-2] = x[1:xs-1,1:ys-1]
        x2[0:xs-2,1:ys-1] = x[1:xs-1,1:ys-1]
        x3[0:xs-2,2:ys]   = x[1:xs-1,1:ys-1]
        x4[1:xs-1,0:ys-2] = x[1:xs-1,1:ys-1]
        x5[1:xs-1,1:ys-1] = x[1:xs-1,1:ys-1]
        x6[1:xs-1,2:ys]   = x[1:xs-1,1:ys-1]
        x7[2:xs,0:ys-2]   = x[1:xs-1,1:ys-1]
        x8[2:xs,1:ys-1]   = x[1:xs-1,1:ys-1]
        x9[2:xs,2:ys]     = x[1:xs-1,1:ys-1]
        x1=x1[1:xs-1,1:ys-1]
        x2=x2[1:xs-1,1:ys-1]
        x3=x3[1:xs-1,1:ys-1]
        x4=x4[1:xs-1,1:ys-1]
        x5=x5[1:xs-1,1:ys-1]
        x6=x6[1:xs-1,1:ys-1]
        x7=x7[1:xs-1,1:ys-1]
        x8=x8[1:xs-1,1:ys-1]
        x9=x9[1:xs-1,1:ys-1]
        dd=[]
        dd.append(x1-x5)
        dd.append(x2-x5)
        dd.append(x3-x5)
        dd.append(x4-x5)
        dd.append(x6-x5)
        dd.append(x7-x5)
        dd.append(x8-x5)
        dd.append(x9-x5)
        map = np.max(dd,axis=0)
        return map

    def getScore(self,x):#x should be double gray image
        if(x.ndim == 3):#color
            x = color.rgb2gray(x)
        map = self.__MLVMap(x)
        xs,ys = map.shape
        xy_number=xs*ys
        vec = map.reshape((xy_number,))
        vec[::-1].sort()#descend
        svec=vec[0:xy_number]
        a=np.arange(xy_number)
        q=np.exp(-0.01*a)
        svec=svec*q
        svec=svec[0:1000]
        return self.__estimateggdparam(svec)