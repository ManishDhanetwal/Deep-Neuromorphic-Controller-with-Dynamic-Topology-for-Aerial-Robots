import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as path_effects

# Data Logger object
class dataLogger(object):
        """docstring for dataLogger"""
        def __init__(self):
            self.ticktime = []
            self.ticks = []
            self.exeTime = []
            self.x_ref = []
            self.y_ref = []
            self.z_ref = []
            self.x_pos = []
            self.y_pos = []
            self.z_pos = []
            self.phi_ref    = []
            self.theta_ref  = []
            self.thrust_ref = []
            self.u_smc   = []
            self.u_nn  = []
            self.nLayer  = []
            self.nNodes  = []
                 
        def appendStateData(self,t,ext,xr,yr,zr,xa,ya,za,phir,thetar,thrustr):
            self.ticktime.append(t)
            self.exeTime.append(ext)
            self.x_ref.append(xr)
            self.y_ref.append(yr)
            self.z_ref.append(zr)
            self.x_pos.append(xa)
            self.y_pos.append(ya)
            self.z_pos.append(za)
            self.phi_ref.append(phir)
            self.theta_ref.append(thetar)
            self.thrust_ref.append(thrustr)

        def appendControlData(self,t,usmc,unn,layer,node):
            self.ticks.append(t)
            self.u_smc.append(usmc)
            self.u_nn.append(unn)
            self.nLayer.append(layer)
            self.nNodes.append(node)

        def saveAll(self):
            timestr = time.strftime("%Y%m%d-%H%M%S")
            print ('Saving all logged data..')
            with open("FlightStateLog"+timestr+".txt", "w") as output:
                output.writelines(map("{};{};{};{};{};{};{};{};{};{};{}\n".format,\
                    self.ticktime,self.exeTime,\
                    self.x_ref, self.y_ref, self.z_ref,\
                    self.x_pos, self.y_pos,self.z_pos, \
                    self.phi_ref, self.theta_ref, self.thrust_ref))
       
        def saveControl(self,axis):
            timestr = time.strftime("%Y%m%d-%H%M%S")
            print ('Saving control logged data..')
            with open(axis+"-ControllerLog"+timestr+".txt", "w") as output:
                output.writelines(map("{};{};{};{};{}\n".format, self.ticks,\
                    self.u_smc, self.u_nn, self.nLayer,self.nNodes))

        def plotFigure(self):
            print ('Plotting the results..')
            exr   = np.array(self.x_ref)
            ex    = np.array(self.x_pos)
            eyr   = np.array(self.y_ref)
            ey    = np.array(self.y_pos)
            ezr   = np.array(self.z_ref)
            ez    = np.array(self.z_pos)
            sqrex = np.multiply((exr-ex),(exr-ex))
            sqrey = np.multiply((eyr-ey),(eyr-ey))
            sqrez = np.multiply((ezr-ez),(ezr-ez))
            rmsex = np.sqrt(sum(sqrex)/np.size(ex))
            rmsey = np.sqrt(sum(sqrey)/np.size(ey))
            rmsez = np.sqrt(sum(sqrez)/np.size(ez))
            print ('RMSE XYZ = ',rmsex,rmsey,rmsez)

            fig = plt.figure("XYZ plot",figsize=(16,9))
            ax1  = plt.subplot2grid((6,6),(0,0),projection='3d', colspan=4,rowspan=6)
            ax1.plot3D(self.x_ref,self.y_ref,self.z_ref, 'k--',label='Target',color='red')
            ax1.plot3D(self.x_pos,self.y_pos,self.z_pos, label='Drone trajectory',color='blue')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_xlim3d(-2,2)
            ax1.set_ylim3d(-2,2)
            ax1.set_zlim3d(0,2)
            ax1.legend()

            ax2 = plt.subplot2grid((6,6),(0,4),colspan=2,rowspan=2)
            ax2.plot(self.ticktime,self.x_ref,self.ticktime,self.x_pos)
            ax2.grid(True)
            ax2.set_xlabel ('time (s)')
            ax2.set_ylabel('X (m)')
            #ax2.text(2, 1, 'RMSE X = %.2f'%rmsex, fontsize=10)

            ax3 = plt.subplot2grid((6,6),(2,4),colspan=2,rowspan=2)
            ax3.plot(self.ticktime,self.y_ref,self.ticktime,self.y_pos)
            ax3.grid(True)
            ax3.set_xlabel ('time (s)')
            ax3.set_ylabel('Y (m)')
            #ax3.text(2, 1, 'RMSE Y = %.2f'%rmsey, fontsize=10)

            ax4 = plt.subplot2grid((6,6),(4,4),colspan=2,rowspan=2)
            ax4.plot(self.ticktime,self.z_ref,self.ticktime,self.z_pos)
            ax4.grid(True)
            ax4.set_xlabel ("time(s)")
            ax4.set_ylabel('Z (m)')
            ax4.text(1, 2, 'RMSE Z = %.2f'%rmsez, fontsize=10)

            plt.show()


        def plotControlData(self):
            print ('Plotting the control signal..')
            plt.figure("Control signal")
            plt.plot(self.ticks,self.u_smc,label='SMC')
            plt.plot(self.ticks,self.u_nn,label='NN')
            plt.grid(True)
            plt.xlabel ("time(s)")
            plt.ylabel('control signal')
            plt.legend()

            plt.figure("evolution")
            plt.plot(self.ticks,self.nLayer,label='layer')
            plt.plot(self.ticks,self.nNodes,label='node')
            plt.grid(True)
            plt.xlabel ("time(s)")
            plt.ylabel('nodes/layers')
            plt.legend()

            plt.show()
