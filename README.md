# Deep-Neuromorphic-Controller-with-Dynamic-Topology-for-Aerial-Robots
Deep Neuromorphic Controller with Dynamic Topology for Aerial Robots code execution setup

Welcome to the simulation world of the published paper on Deep neuromorphic controller . 

Here is a list of simple commands and steps to be able to execute the codes and see results for further research.

##INSTALL PACKAGES
1. Clean Linux machine running ubuntu 16.04 . * Any sort of ros or px4 which is of different version might not work correctly with this implementation as such use a from a clean installation. 
2.Run the following commands in order : 
  *(Any simple errors, please refer to the compiler suggestion ) i have tried to cover all here as i was able to reproduce the results.
  ##
  1. wget https://raw.githubusercontent.com/PX4/Devguide/master/build_scripts/ubuntu_sim_ros_melodic.shls\
  2. source ubuntu_sim_ros_melodic.sh
 
  ## After this make a new folder at desktop and download px4 and run the script:
    1.  git clone https://github.com/PX4/PX4-Autopilot
    2.  cd PX4/-Autopilot
    3.  make p4_sitl gazebo
 
  ## Now you will have numerous error on each run , so just run the following command to save time:
      1.  pip3 install –user empy
      2.  pip3 install –user packaging
      3.  pip3 install –user toml
      4.  pip3 install –user numpy
      5.  pip3 install -user jinja2
      6.  sudo apt install libgstreamer1.0-dev
      7.  sudo apt install gstreamer1.0-plugins-good
      8.  sudo apt install gstreamer1.0-plugins-bad
      9.  sudo apt install gstreamer1.0-plugins-ugly
      The px4 should have run succefully now ! if not just run this once 
      10. Sudo apt-get update
      11. Sudo apt-upgrade 
  ##  You should have the px4 running a simulation now. Close it
    4.  We will use Gazebo with ROS wrapper for this project
    5.  cd px4/PX4-Autopilot
    6.  DONT_RUN=1 make px4_sitl_default gazebo
    7.  source Tools/setup_gazebo.bash $(pwd) $(pwd)/build/px4_sitl_default
    8.  export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)
    9.  export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)/Tools/sitl_gazebo
    10.  roslaunch px4 mavros_posix_sitl.launch
  3.  Now the px4 would hv connected to the simulator gazebo on TCP port 4560
  4.  Choose a script from the src folder and just run it with python ./ypur_choice_of_script


