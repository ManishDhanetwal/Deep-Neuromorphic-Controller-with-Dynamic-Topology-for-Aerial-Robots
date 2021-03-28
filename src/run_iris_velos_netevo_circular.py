#!/usr/bin/env python

# ROS python API
import rospy

# 3D point & Stamped Pose msgs
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion, Vector3
from geometry_msgs.msg import TwistStamped

# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *

#import pymavlink
#from std_msgs.msg import Header
#from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion

import time
import math
import numpy as np

from termcolor import colored, cprint
import copy
import pdb

from evonn_corr import *
from px4api import *
from datalogger_new import *
from tools import *
from trajectories import *


# Main function
def main():

# INITIALIZE ADAPTIVE NETWORK PARAMETERS:
    N_INPUT       = 4                   # Number of input network
    N_OUTPUT      = 1                   # Number of output network
    INIT_NODE     = 4                   # Number of node(s) for initial structure
    INIT_LAYER    = 2                   # Number of initial layer (minimum 2, for 1 hidden and 1 output)
    N_WINDOW      = 300                 # Sliding Window Size
    EVAL_WINDOW   = 3                   # Evaluation Window for layer Adaptation
    DELTA         = 0.05                # Confidence Level for layer Adaptation (0.05 = 95%)
    ETA           = 0.0001              # Minimum allowable value if divided by zero
    FORGET_FACTOR = 0.98                # Forgetting Factor of Recursive Calculation
    #EXP_DECAY     = 1                  # Learning Rate decay factor
    #LEARNING_RATE = 0.00004             # Network optimization learning rate
    #LEARNING_RATE = 0.01             # Network optimization learning rate
    # LEARNING_RATE_X = 0.02             # Network optimization learning rate 0.00085
    # LEARNING_RATE_Y = 0.02             # Network optimization learning rate 0.00085
    # LEARNING_RATE_Z = 0.05             # Network optimization learning rate

    LEARNING_RATE_X = 0.05             # Network optimization learning rate 0.00085
    LEARNING_RATE_Y = 0.05             # Network optimization learning rate 0.00085
    LEARNING_RATE_Z = 0.5             # Network optimization learning rate

    WEIGHT_DECAY  = 0.000125            # Regularization weight decay factor
    #WEIGHT_DECAY  = 0.0                 # Regularization weight decay factor
    #SLIDING_WINDOW = 35

# INITIALIZE SYSTEM AND SIMULATION PARAMETERS:
    IRIS_THRUST   = 0.5629                 # Nominal thrust for IRIS Quadrotor
    RATE          = 25.0                    # Sampling Frequency (Hz)
    # PID_GAIN_X    = [0.25,0.0,0.02]         # PID Gain Parameter [KP,KI,KD] axis-X
    # PID_GAIN_Y    = [0.25,0.0,0.02]         # PID Gain Parameter [KP,KI,KD] axis-Y
    PID_GAIN_X    = [0.45,0.0,0.002]         # PID Gain Parameter [KP,KI,KD] axis-X
    PID_GAIN_Y    = [0.45,0.0,0.002]         # PID Gain Parameter [KP,KI,KD] axis-Y
    #PID_GAIN_Z    = [0.013,0.0004,0.2]      # PID Gain Parameter [KP,KI,KD] axis-Z
    PID_GAIN_Z    = [0.85,0.0,0.0001]         # PID Gain Parameter [KP,KI,KD] axis-Z
    SIM_TIME      = 200                     # Simulation time duration (s)
#--------------------------------------------------------------------------------

# Initial conditions of UAV system
    xref     = 0.0
    yref     = 0.0
    zref     = 1.0
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
    trajectory = Trajectory()
    meanErrX = recursiveMean()
    meanErrY = recursiveMean()
    meanErrZ = recursiveMean()

# Data Logger object
    logData = dataLogger('flight')
    xconLog = dataLogger('X')
    yconLog = dataLogger('Y')
    zconLog = dataLogger('Z')

# Initiate node and subscriber
    rospy.init_node('setpoint_node', anonymous=True)
    rate  = rospy.Rate(RATE)        # ROS loop rate, [Hz]

    # Subscribe to drone state
    rospy.Subscriber('mavros/state', State, uav.stateCb)

    # Subscribe to drone's local position
    rospy.Subscriber('mavros/local_position/pose', PoseStamped, uav.posCb)

    # Setpoint publisher
    velocity_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)

# Velocity messages init
    vel = TwistStamped()
    vel.twist.linear.x = 0.0
    vel.twist.linear.y = 0.0
    vel.twist.linear.z = 0.0

# Arming the UAV --> no auto arming, please comment out line 140
    text = colored("Ready for Flight..Press Enter to continue...", 'green', attrs=['reverse', 'blink'])
    raw_input(text)

    k=0
    while not uav.state.armed:
        modes.setArm()
        rate.sleep()
        k += 1

        if k % 10 == 0:
            text = colored('Waiting for arming..', 'yellow')
            print text

        if k > 500:
            text = colored('Arming timeout..', 'red', attrs=['reverse', 'blink'])
            print text
            break
    
# Switch to OFFBOARD after send few setpoint messages
    text = colored("Vehicle Armed!! Press Enter to switch OFFBOARD...", 'green', attrs=['reverse', 'blink'])
    raw_input(text)

    while not uav.state.armed:
        modes.setArm()
        rate.sleep()
        text = colored('Vehicle is arming..', 'yellow')
        print text

    rate.sleep()
    k=0
    while k<10:
        velocity_pub.publish(vel)
        rate.sleep()
        k = k+1
    modes.setOffboardMode()
    text = colored('Now in OFFBOARD Mode..', 'blue', attrs=['reverse', 'blink'])
    print text
    
# ROS Main loop::
    k = 0
    while not rospy.is_shutdown():
        start = time.time()
        rate.sleep()

        # Setpoint generator:
        # Take off with altitude Z = 1 m
        if runtime <= 15:
            xref = 0.0
            yref = 0.0
            zref = 0.3+ trajectory.linear(0.7,dt,15)
            
            if runtime == 15:
                trajectory.reset()
        
        # Sinusoidal Z trajectory after 20 s 
        # if runtime > 20:
        #     xref = 0.0
        #     yref = 0.0
        #     zref = 1.0 + trajectory.sinusoid(0.25,dt,20)
        
        if runtime > 15 and runtime <= 25:
            xref = trajectory.linear(1.0,dt,10)
            yref = 0.0
            zref = 1.0
            
            if runtime == 25:
                trajectory.reset()

        # Circle XY only
        if runtime > 25:
            xref,yref = trajectory.circle(1.0,dt,20)
            zref = 1.0


        # Circle XY untill 150 s
        # if runtime > 25 and runtime <= 150:
        #     xref,yref = trajectory.circle(1.0,dt,20)
        #     zref = 1.0

        #     if runtime == 150:
        #         trajectory.reset()

        # # Eight shape XY after 150 s
        # if runtime > 150:
        #     xref,yref = trajectory.eight(1.5,0.75,dt,30,15)
        #     zref = 1.3

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
        vz  = Zcon.controlUpdate(zref)      # Additional Thrust Z
        
        vx = limiter(vx,1)
        vy = limiter(vy,1)
        vz = limiter(vz,1)

        # Publish the control signal
        vel.twist.linear.x = vx
        vel.twist.linear.y = vy
        vel.twist.linear.z = vz
        velocity_pub.publish(vel)
        
        # NN Learning stage
        # if meanErrX.updateMean(abs(errx),0.99) > 0.05:
        #     Xcon.optimize(LEARNING_RATE_X,WEIGHT_DECAY)

        # if meanErrY.updateMean(abs(erry),0.99) > 0.05:
        #     Ycon.optimize(LEARNING_RATE_Y,WEIGHT_DECAY)

        # if meanErrZ.updateMean(abs(errz),0.99) > 0.05:
        #     Zcon.optimize(LEARNING_RATE_Z,WEIGHT_DECAY)

        Xcon.optimize(LEARNING_RATE_X,WEIGHT_DECAY)

        Ycon.optimize(LEARNING_RATE_Y,WEIGHT_DECAY)

        Zcon.optimize(LEARNING_RATE_Z,WEIGHT_DECAY)

        # Adjust the number of nodes in the latest layer (Network Width)
        Xcon.adjustWidth(FORGET_FACTOR,N_WINDOW)
        Ycon.adjustWidth(FORGET_FACTOR,N_WINDOW)
        Zcon.adjustWidth(FORGET_FACTOR,N_WINDOW)

        # # Adjust the number of layer (Network Depth)
        Xcon.adjustDepth(ETA,DELTA,N_WINDOW,EVAL_WINDOW)
        Ycon.adjustDepth(ETA,DELTA,N_WINDOW,EVAL_WINDOW)
        Zcon.adjustDepth(ETA,DELTA,N_WINDOW,EVAL_WINDOW)

        #euler = euler_from_quaternion(uav.q)
        # Print all states
        print 'Xr,Yr,Zr    = ',xref,yref,zref
        print 'X, Y, Z     = ',uav.local_pos.x,uav.local_pos.y,uav.local_pos.z
        print 'ex, ey, ez  = ',errx,erry,errz
        print 'vx,vy,vz    = ',vx,vy,vz
        #print 'att angles  = ',euler
        print 'Nodes X Y Z = ',Xcon.nNodes,Ycon.nNodes, Zcon.nNodes
        print 'Layer X Y Z = ',Xcon.nLayer,Ycon.nLayer, Zcon.nLayer
        # print Ycon.netStruct[0].linear.weight.data
        # print Ycon.netStruct[1].linear.weight.data
        print

        k+=1
        runtime  = k*dt
        execTime = time.time()-start
        print 'Runtime    = ',runtime
        print 'Exec time  = ',execTime
        print

        # Append logged Data
        logData.appendStateData(runtime,execTime,xref,yref,zref,xpos,ypos,zpos,vx,vy,vz)
        xconLog.appendControlData(runtime,Xcon.us,Xcon.un,Xcon.nLayer,Xcon.nNodes)
        yconLog.appendControlData(runtime,Ycon.us,Ycon.un,Ycon.nLayer,Ycon.nNodes)
        zconLog.appendControlData(runtime,Zcon.us,Zcon.un,Zcon.nLayer,Zcon.nNodes)

        # Save logged Data
        logData.saveStateData(runtime,execTime,xref,yref,zref,xpos,ypos,zpos,vx,vy,vz)
        xconLog.saveControlData(runtime,Xcon.us,Xcon.un,Xcon.nLayer,Xcon.nNodes)
        yconLog.saveControlData(runtime,Ycon.us,Ycon.un,Ycon.nLayer,Ycon.nNodes)
        zconLog.saveControlData(runtime,Zcon.us,Zcon.un,Zcon.nLayer,Zcon.nNodes)

        # Break condition
        if runtime > SIM_TIME:
        	# Set auto land mode
            modes.setRTLMode()

            text = colored('Auto landing mode now..', 'blue', attrs=['reverse', 'blink'])
            print text
            text = colored('Now saving all logged data..', 'yellow', attrs=['reverse', 'blink'])
            print text

            # Closing the saved log files
            logData.logFile.close()
            xconLog.logFile.close()
            yconLog.logFile.close()
            zconLog.logFile.close()

            # Plotting the results
            logData.plotFigure()
            xconLog.plotControlData()
            yconLog.plotControlData()
            zconLog.plotControlData()
            text = colored('Mission Complete!!', 'green', attrs=['reverse', 'blink'])
            print text

            break

# Call main function
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
