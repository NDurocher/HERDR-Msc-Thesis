#!/bin/bash

cd /home/nathan/HERDR/Test/controllers/Hircus
python3 convert.py

cd ..

cd /opt/intel/openvino_2021.4.752/deployment_tools/model_optimizer
python3 mo_onnx.py --input_model /home/nathan/HERDR/Test/controllers/Hircus/Herdr.onnx --output_dir /home/nathan/HERDR/Test/controllers/Hircus --input "img[50 3 240 320],actions[50 20 2]" 
