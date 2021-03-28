#!/usr/bin/env python

# ROS python API
import rospy

# 3D point & Stamped Pose msgs
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion, Vector3

# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *

#import pymavlink
from mavros_msgs.msg import AttitudeTarget
#from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion

import time
import math
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patheffects as path_effects
from termcolor import colored, cprint
import copy
import pdb

# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F
# import torch.utils.data as Data

from evonn import *
from px4api import *
from datalogger import *
from tools import *


# Main function
def main():

# INITIALIZE ADAPTIVE NETWORK PARAMETERS:
    N_INPUT       = 3                   # Number of input network
    N_OUTPUT      = 1                   # Number of output network
    INIT_NODE     = 1                   # Number of node(s) for initial structure
    INIT_LAYER    = 2                   # Number of initial layer (minimum 2, for 1 hidden and 1 output)
    N_WINDOW      = 500                 # Sliding Window Size
    EVAL_WINDOW   = 3                   # Evaluation Window for layer Adaptation
    DELTA         = 0.05                # Confidence Level for layer Adaptation (0.05 = 95%)
    ETA           = 0.0001              # Minimum allowable value if divided by zero
    FORGET_FACTOR = 0.98                # Forgetting Factor of Recursive Calculation
    #EXP_DECAY     = 1                  # Learning Rate decay factor
    #LEARNING_RATE = 0.00005             # Network optimization learning rate
    LEARNING_RATE = 0.00002/2            # Network optimization learning rate
    WEIGHT_DECAY  = 0.000125/5            # Regularization weight decay factor
    #WEIGHT_DECAY  = 0.0                 # Regularization weight decay factor

# INITIALIZE SYSTEM AND SIMULATION PARAMETERS:
    IRIS_THRUST   = 0.5629                  # Nominal thrust for IRIS Quadrotor
    RATE          = 30.0                    # Sampling Frequency (Hz)
    # PID_GAIN_X    = [0.25,0.0,0.02]         # PID Gain Parameter [KP,KI,KD] axis-X
    # PID_GAIN_Y    = [0.25,0.0,0.02]         # PID Gain Parameter [KP,KI,KD] axis-Y
    PID_GAIN_X    = [0.15,0.0,0.004]         # PID Gain Parameter [KP,KI,KD] axis-X
    PID_GAIN_Y    = [0.15,0.0,0.004]         # PID Gain Parameter [KP,KI,KD] axis-Y
    #PID_GAIN_Z    = [0.013,0.0004,0.2]      # PID Gain Parameter [KP,KI,KD] axis-Z
    PID_GAIN_Z    = [0.013,0.0,0.2]         # PID Gain Parameter [KP,KI,KD] axis-Z
    SIM_TIME      = 200                     # Simulation time duration (s)

# Initial conditions of UAV system
    xref     = 0.0
    yref     = 0.0
    zref     = 2.0
    interrx  = 0.0
    interry  = 0.0
    interrz  = 0.0
    errlastx = 0.0
    errlasty = 0.0
    errlastz = 0.0
    runtime  = 0.0

# Ignite the Evolving NN Controller
    Xcon = NetEvo(N_INPUT,N_OUTPUT,INIT_NODE)
    Ycon = NetEvo(N_INPUT,N_OUTPUT,INIT_NODE)
    Zcon = NetEvo(N_INPUT,N_OUTPUT,INIT_NODE)

    if INIT_LAYER > 2:
        pdb.set_trace()
        for i in range(INIT_LAYER-2):
            Xcon.addhiddenlayer()
            Ycon.addhiddenlayer()
            Zcon.addhiddenlayer()

    dt      = 1/RATE
    Xcon.dt = dt
    Ycon.dt = dt
    Zcon.dt = dt
    Xcon.smcpar = PID_GAIN_X
    Ycon.smcpar = PID_GAIN_Y
    Zcon.smcpar = PID_GAIN_Z

# PX4 API object
    modes = fcuModes()              # Flight mode object
    uav   = uavCommand()            # UAV command object

# Data Logger object
    logData = dataLogger()
    xconLog = dataLogger()
    yconLog = dataLogger()
    zconLog = dataLogger()

# Initiate node and subscriber
    rospy.init_node('setpoint_node', anonymous=True)
    rate  = rospy.Rate(RATE)        # ROS loop rate, [Hz]

    # Subscribe to drone state
    rospy.Subscriber('mavros/state', State, uav.stateCb)

    # Subscribe to drone's local position
    #rospy.Subscriber('mavros/local_position/pose', PoseStamped, uav.posCb)
    rospy.Subscriber('mavros/mocap/pose', PoseStamped, uav.posCb)

    # Setpoint publisher
    att_sp_pub = rospy.Publisher('mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)

# Initiate Attitude messages
    att             = AttitudeTarget()
    att.body_rate   = Vector3()
    att.orientation = Quaternion(*quaternion_from_euler(0.0, 0.0,0.0))
    att.thrust      = IRIS_THRUST
    att.type_mask   = 7 

# Arming the UAV --> no auto arming
    text = colored("Ready for Arming..Press Enter to continue...", 'green', attrs=['reverse', 'blink'])
    raw_input(text)
    while not uav.state.armed:
        modes.setArm()
        rate.sleep()
        text = colored('Vehicle is arming..', 'yellow')
        print (text)
    
# Switch to OFFBOARD after send few setpoint messages
    text = colored("Vehicle Armed!! Press Enter to switch OFFBOARD...", 'green', attrs=['reverse', 'blink'])
    raw_input(text)

    while not uav.state.armed:
        modes.setArm()
        rate.sleep()
        text = colored('Vehicle is arming..', 'yellow')
        print (text)

    rate.sleep()
    k=0
    while k<10:
        att_sp_pub.publish(att)
        rate.sleep()
        k = k+1
    modes.setOffboardMode()
    text = colored('Now in OFFBOARD Mode..', 'blue', attrs=['reverse', 'blink'])
    print (text)
    
# ROS Main loop::
    k = 0
    while not rospy.is_shutdown():
        start = time.time()
        rate.sleep()

        # Setpoint generator
        if runtime <= 20:
            xref=0.0
            yref=0.0
            zref=runtime* 1.5/20.0
        if runtime > 20 and runtime <=30:
            xref = (runtime-20)* 1.0/10.0
            #xref = (runtime-20)* 5.0/10.0
            yref = 0.0
            zref = 1.5
        if runtime>30:
            xref = 0.0+1.0*math.cos(math.pi*(runtime-30)/20.0)
            yref = 0.0+1.0*math.sin(math.pi*(runtime-30)/20.0)
            #xref = 0.0+5.0*math.cos(math.pi*(runtime-30)/60.0)
            #yref = 0.0+5.0*math.sin(math.pi*(runtime-30)/60.0)
            zref = 1.5 

        # Adjust the number of nodes in the latest layer (Network Width)
        Xcon.adjustWidth(FORGET_FACTOR,N_WINDOW)
        Ycon.adjustWidth(FORGET_FACTOR,N_WINDOW)
        #Zcon.adjustWidth(FORGET_FACTOR,N_WINDOW)

        # Adjust the number of layer (Network Depth)
        Xcon.adjustDepth(ETA,DELTA,N_WINDOW,EVAL_WINDOW)
        Ycon.adjustDepth(ETA,DELTA,N_WINDOW,EVAL_WINDOW)
        #Zcon.adjustDepth(ETA,DELTA,N_WINDOW,EVAL_WINDOW)

        # update current position
        xpos = uav.local_pos.x
        ypos = uav.local_pos.y
        zpos = uav.local_pos.z
        
        # calculate errors,delta errors, and integral errors
        errx,derrx,interrx = Xcon.calculateError(xref,xpos)
        erry,derry,interry = Ycon.calculateError(yref,ypos)
        errz,derrz,interrz = Zcon.calculateError(zref,zpos)

        # PID Controller equations
        #theta_ref = 0.04*errx+0.0005*interrx+0.01*derrx
        #phi_ref   = 0.04*erry+0.0005*interry+0.01*derry
        # vx = PID_GAIN_X[0]*errx+PID_GAIN_X[1]*interrx+PID_GAIN_X[2]*derrx
        # vy = PID_GAIN_Y[0]*erry+PID_GAIN_Y[1]*interry+PID_GAIN_Y[2]*derry
        #thrust_ref = IRIS_THRUST+PID_GAIN_Z[0]*errz+PID_GAIN_Z[1]*interrz+PID_GAIN_Z[2]*derrz

        # SMC + NN controller
        vx  = Xcon.controlUpdate(xref)      # Velocity X
        vy  = Ycon.controlUpdate(yref)      # Velocity Y
        thz = Zcon.controlUpdate(zref)      # Additional Thrust Z
        
        euler = euler_from_quaternion(uav.q)
        psi   = euler[2]
        phi_ref,theta_ref = transform_control_signal(vx,vy,psi)

        thrust_ref = IRIS_THRUST + thz

        # control signal limiter
        #phi_ref    = limiter(-1*phi_ref,0.2)
        phi_ref    = limiter(-1*phi_ref,0.25)
        theta_ref  = limiter(-1*theta_ref,0.25)
        att.thrust = limiter(thrust_ref,1)

        # Phi, Theta and Psi quaternion transformation
        att.orientation = Quaternion(*quaternion_from_euler(phi_ref, theta_ref, 0.0))

        # Publish the control signal
        att_sp_pub.publish(att)
        
        # NN Learning stage
        Xcon.optimize(LEARNING_RATE,WEIGHT_DECAY)
        Ycon.optimize(LEARNING_RATE,WEIGHT_DECAY)
        Zcon.optimize(LEARNING_RATE,WEIGHT_DECAY)

        # Print all states
        print ('Xr,Yr,Zr    = ',xref,yref,zref)
        print ('X, Y, Z     = ',uav.local_pos.x,uav.local_pos.y,uav.local_pos.z)
        print ('phi, theta  = ',phi_ref,theta_ref)
        print ('att angles  = ',euler)
        print ('Thrust      = ',att.thrust)
        print ('Nodes X Y Z = ',Xcon.nNodes,Ycon.nNodes, Zcon.nNodes)
        print ('Layer X Y Z = ',Xcon.nLayer,Ycon.nLayer, Zcon.nLayer)
        
        k+=1
        runtime = k*dt

        # Append Data Logger
        logData.appendStateData(runtime,xref,yref,zref,xpos,ypos,zpos,phi_ref,theta_ref,thrust_ref)
        xconLog.appendControlData(runtime,Xcon.us,Xcon.un,Xcon.u,Xcon.nNodes)
        yconLog.appendControlData(runtime,Ycon.us,Ycon.un,Ycon.u,Ycon.nNodes)
        zconLog.appendControlData(runtime,Zcon.us,Zcon.un,Zcon.u,Zcon.nNodes)

        print ('Runtime    = ',runtime)
        print ('Exec time  = ', time.time()-start)
        print ('')
        # Break condition
        if runtime > SIM_TIME:
            modes.setRTLMode()

            text = colored('Auto landing mode now..', 'blue', attrs=['reverse', 'blink'])
            print (text)
            text = colored('Now saving all logged data..', 'yellow', attrs=['reverse', 'blink'])
            print (text)
            
            logData.plotFigure()
            xconLog.plotControlData()
            yconLog.plotControlData()
            zconLog.plotControlData()
            
            text = colored('Mission Complete!!', 'green', attrs=['reverse', 'blink'])
            print (text)
            break

# Call main function
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
