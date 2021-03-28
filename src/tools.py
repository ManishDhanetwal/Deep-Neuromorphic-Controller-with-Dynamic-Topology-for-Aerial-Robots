import math
import numpy as np

# Control signal transformation
def transform_control_signal(vx,vy,psi,vz=0.0,g=9.81,m=1.6):
    
    thrust_ref = m*math.sqrt(1*vx**2+1*vy**2+(vz+g)**2)
    # phi_ref    = asin((1/g)*(vx*sin(psi)-vy*cos(psi)))
    # theta_ref  = atan((1/g)*(vx*cos(psi)+vy*sin(psi)))

    theta_r  = 1*math.atan((1/(vz+g))*(vx*math.cos(psi)+vy*math.sin(psi)))
    phi_r   = 1*math.asin((m/thrust_ref)*(vx*math.sin(psi)-vy*math.cos(psi)))

    return phi_r,theta_r

# Recursive mean and Bias
class recursiveMeanVar(object):
    def __init__(self):
        self.weight = 1.0
        self.mean = 0.0
        self.var = 0.0
        self.xk = [0.0]

    def updateMean(self,newData,forgettingFactor,varWindow):
        self.xk.append(newData)
        
        if len(self.xk)>varWindow:
            self.xk.pop(0)

        self.weight = forgettingFactor*self.weight + 1
        self.mean   = (1-1/self.weight)*self.mean + (1/self.weight)*newData

        self.vn     = 2*forgettingFactor*(1-(forgettingFactor**(varWindow-1)))/(1-forgettingFactor)*(1+forgettingFactor)
        
        sumvar = 0.0
        for k in range(len(self.xk)):
            sumvar+=(forgettingFactor**(len(self.xk)-(k+1)))*((self.mean-self.xk[k])**2)
        
        self.var = 1/self.vn*sumvar

        return self.mean, self.var

# Saturation limit function
def limiter(x,lim):
    if x>=lim:
        yout=lim
    elif x<=-1*lim:
        yout=-1*lim
    else:
        yout=x
    return yout


class recursiveMean(object):
    def __init__(self):
        self.reset()

    def updateMean(self,newData,forgettingFactor):
        self.weight = forgettingFactor*self.weight + 1
        self.mean   = (1-1/self.weight)*self.mean + (1/self.weight)*newData

        return self.mean

    def reset(self):
        self.weight = 1.0
        self.mean = 0.0

# Recursive correlation
class recCorr(object):
    def __init__(self):
        self.weight = 1.0
        self.meanx = 0.0
        self.varx = 0.0
        self.meany = 0.0
        self.vary = 0.0
        self.sxy = 0.0

        self.xk = []
        self.yk = []

    def updateCorr(self,x,y,forgettingFactor,varWindow):
        self.xk.append(x)
        self.yk.append(y)
        nData = len(self.xk)
        
        if nData>varWindow:
            self.xk.pop(0)
            self.yk.pop(0)
            nData = nData-1


        self.weight = forgettingFactor*self.weight + 1

        self.meanx  = (1-1/self.weight)*self.meanx + (1/self.weight)*x
        self.meany  = (1-1/self.weight)*self.meany + (1/self.weight)*y

        self.vn     = 2*forgettingFactor*(1-(forgettingFactor**(varWindow-1)))/(1-forgettingFactor)*(1+forgettingFactor)
        
        sumVarx = 0.0
        sumVary = 0.0
        sXY = 0.0

        for k in range(nData):
            sumVarx += (forgettingFactor**(nData-(k+1)))*((self.meanx-self.xk[k])**2)
            sumVary += (forgettingFactor**(nData-(k+1)))*((self.meany-self.yk[k])**2)
            sXY  += (forgettingFactor**(nData-(k+1)))*((self.yk[k]-self.meany)*(self.xk[k]-self.meanx))

        self.varx = 1/self.vn*sumVarx
        self.vary = 1/self.vn*sumVary

        self.stdx = self.varx.sqrt()
        self.stdy = self.vary.sqrt()

        self.cov  = 1/self.vn*sXY
        self.corr = self.cov/(self.stdx*self.stdy)

        return self.corr