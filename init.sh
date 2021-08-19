#!/bin/bash

#conda create --prefix /home/tione/notebook/envs/tf_1 -y python=3.6 ipykernel cudatoolkit=10.0 cudnn -c conda-forge
conda create --prefix /home/tione/notebook/envs/tf_1 -y python=3.6 ipykernel cudatoolkit=10.0 cudnn -c http://mirrors.tencentyun.com/pypi/simple

source activate /home/tione/notebook/envs/tf_1

pip install -r requirements.txt