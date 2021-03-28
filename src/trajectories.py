import numpy as np
import math

class Trajectory(object):
    def __init__(self,):
        self.reset()

    def reset(self):
        self.t = 0

    def circle(self,r,dt,period):
        self.t += 1
        x  = r * np.cos(2*math.pi*(self.t*dt)/period)
        y  = r * np.sin(2*math.pi*(self.t*dt)/period)

        return x,y

    def linear(self,target,dt,duration):
        self.t += 1
        if self.t*dt <= duration:
        	point = self.t*dt*target/duration
        else:
        	point = target

        return point

    def sinusoid(self,amp,dt,period):
        self.t += 1
        point  =  amp* np.sin(2*math.pi*(self.t*dt)/period)

        return point

    def eight(self,ampx,ampy,dt,tx,ty):
        self.t += 1
        x  = ampx * np.cos(2*math.pi*(self.t*dt)/tx)
        y  = ampy * np.sin(2*math.pi*(self.t*dt)/ty)

        return x,y

    def pulse(self,amp,dt,period):
        self.t += 1
        tim=self.t*dt

        if tim <= period/4.0:
            point = tim*amp/period*4.0

        elif tim > period/4.0 and tim <= period/2.0:
            point = amp

        elif tim > period/2.0 and tim <= period*3.0/4.0:
            point = amp - (tim-(period/2.0))*amp/period*4.0

        elif tim > period*3.0/4.0 and tim <= period:
            point = 0.0

            if tim==period:
                self.reset()

        return point

## EXPERIMENT

# xt=Trajectory()
# dt=0.1
# for i in range(100):
# 	x=1+xt.pulse(3,0.1,10)
# 	print 'x = ',x