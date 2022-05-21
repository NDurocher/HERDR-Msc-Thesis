ARG ROS_DISTRO=noetic
ARG FROM_IMAGE=ros:$ROS_DISTRO-ros-base
ARG OVERLAY_WS=/opt/capra/overlay_ws
ARG ROS_SETUP=/opt/ros/$ROS_DISTRO/setup.sh

# MULTI-STAGE FOR CACHING
FROM $FROM_IMAGE AS cacher

# copy overlay source   
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS/src
COPY ./ros-bridge/ carla-ros-bridge/
COPY ./carla-0.9.12-py3.7-linux-x86_64.egg .

# copy manifests for caching
# WORKDIR /opt
# RUN mkdir -p /tmp/opt && \
#   find ./ -name "package.xml" | \
#   xargs cp --parents -t /tmp/opt \
#   || true

# MULTI-STAGE FOR BUILDING
FROM $FROM_IMAGE AS builder
ARG DEBIAN_FRONTEND=noninteractive

# install overlay dependencies
# ARG OVERLAY_WS
# WORKDIR $OVERLAY_WS
# ARG ROS_DISTRO
# install ros
#RUN apt-get update && apt-get install -y lsb-release && apt-get clean all \
# && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
# && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && apt-get update && apt-get install  -q -y --no-install-recommends ros-$ROS_DISTRO-desktop-full && apt-get install  -q -y --no-install-recommends python3-rosdep && rosdep init && rosdep update

# install CI dependencies
ARG ROS_DISTRO
RUN apt-get update && apt-get install -q -y --no-install-recommends\
  ccache \
  lcov \
  git \
  python \
  net-tools \ 
  iputils-ping \
  python3-pip \
  python-numpy \
  python-yaml \  
  libeigen3-dev \
  libnlopt-dev \
  build-essential \
  unzip \
  nvidia-cuda-toolkit \ 
  g++ \
  nano \
  ros-$ROS_DISTRO-vision-msgs \
  ros-$ROS_DISTRO-camera-info-manager \
  ros-$ROS_DISTRO-cv-bridge \
  ros-$ROS_DISTRO-pcl-ros \
  ros-$ROS_DISTRO-tf-conversions \
  ros-$ROS_DISTRO-nlopt \
  ros-$ROS_DISTRO-actionlib-msgs \
  && pip3 install scipy scikit-learn==0.22.2\
  && rm -rf /var/lib/apt/lists/*
  
# RUN pip install scikit-learn typing numpy 
RUN pip3 install scikit-learn typing numpy 
# RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# RUN pip3 install carla==0.9.12
# tensorflow-gpu

# install tensorflow cc
RUN apt-get update && apt-get install -q -y --no-install-recommends cmake curl g++-7 git python3-dev python3-numpy sudo wget


RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg && \
mv bazel.gpg /etc/apt/trusted.gpg.d/ && \
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

RUN apt-get update && apt-get install -q -y --no-install-recommends bazel bazel-3.7.2


ARG ROS_SETUP
ARG ROS_DISTRO
RUN . $ROS_SETUP && \
  apt-get update && \
  rosdep update && \
  rosdep install -q -y \
  --from-paths src \
  --rosdistro=$ROS_DISTRO \
  --ignore-src ; exit 0\
  && rm -rf /var/lib/apt/lists/*

COPY --from=cacher $OVERLAY_WS ./

RUN rm /usr/bin/python
RUN ln -s /usr/bin/python3 /usr/bin/python

ARG OVERLAY_WS
ARG ROS_DISTRO
WORKDIR $OVERLAY_WS
RUN . $ROS_SETUP && \
    rosdep update && \
    rosdep install -q -y \
    --from-paths src \
    --rosdistro=$ROS_DISTRO \
    --ignore-src -r ; exit 0\
  && rm -rf /var/lib/apt/lists/*
RUN . $ROS_SETUP && catkin_make


COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN ["chmod", "+x", "/ros_entrypoint.sh"]
ENTRYPOINT [ "/ros_entrypoint.sh" ] 
# CMD ["python3", "src/carla-ros-bridge/carla_ros_bridge/src/carla_ros_bridge/bridge.py"]
# CMD ["roslaunch", "carla_twist_to_control", "carla_twist_to_control.launch"]
CMD ["roslaunch", "carla_ros_bridge", "carla_ros_bridge.launch", "--wait"]
