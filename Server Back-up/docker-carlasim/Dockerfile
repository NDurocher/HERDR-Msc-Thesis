FROM carlasim/carla:0.9.12
USER root
RUN apt-get update && apt-get install -y xdg-user-dirs xdg-utils && apt-get clean
RUN apt-get update && apt-get install -y python3 libjpeg-turbo8 libtiff5 libpng16-16 nano curl sudo locales && apt-get clean
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN export XDG_RUNTIME_DIR=/home/tmp && export RUNLEVEL=3
USER carla