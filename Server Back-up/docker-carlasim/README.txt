To start container rn:
docker run -it --rm --privileged --gpus all --name CARLA --expose 2000 --net=host -e DISPLAY=$DISPLAY -e --runtime=nvidia -v /tmp/.X11-unix:/tmp/.X11-unix:rw carla/working ./CarlaUE4.sh -RenderOffScreen

--runtime flag was important

To test complete Herdr algorithm:
1. Start this docker (carlasim)
2. Start roscore and navigation from carpa-urban-cleaning
        2b. Navigation starts the carla environment and spawns "ego" vehicle (Carla_Run_Herdr.py)
3. Start ros_bridge & twist-to-carla_command nodes (docker-carla-ros-bridge)

NOTE:
On exit/termination of navigation, 1. and 3. must be restarted as well