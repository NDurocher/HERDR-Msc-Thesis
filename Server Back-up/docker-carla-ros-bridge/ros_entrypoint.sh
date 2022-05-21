#!/bin/bash
set -e 
# export CARLA_ROOT=/opt/carla-simulator
export PYTHONPATH=$PYTHONPATH:/opt/capra/overlay_ws/src/carla-0.9.12-py3.7-linux-x86_64.egg
# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/opt/capra/overlay_ws/devel/setup.bash"
exec "$@"