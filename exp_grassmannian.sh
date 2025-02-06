#!/bin/bash

#---- GyroGr ----
path=/data #change this to your data folder

### Experiments on GyroBN under 1Block architeture
[ $? -eq 0 ] && python GyroBNGr.py -m\
  dataset.path=$path\
  nnet=GyroGr\
  nnet.model.architecture=[[93,93]]\
  nnet.model.is_pooling=True\
  nnet.model.is_final_pooling=True\
  nnet.BN_param.is_bn=True,False\
  nnet.BN_param.bn_type=GyroGrBN\
  hydra.job_logging.handlers.file.filename=1Block.log\
  hydra.run.dir=./outputs/HDM05\
  hydra.sweep.dir=./outputs/HDM05

### Experiments on GyroBN under 2Block architeture
[ $? -eq 0 ] && python GyroBNGr.py -m\
  dataset.path=$path\
  nnet=GyroGr\
  nnet.model.architecture=[[93,93],[47,47]]\
  nnet.model.is_pooling=True\
  nnet.model.is_final_pooling=False\
  nnet.BN_param.is_bn=True,False\
  nnet.BN_param.bn_type=GyroGrBN\
  hydra.job_logging.handlers.file.filename=2Block.log\
  hydra.run.dir=./outputs/HDM05\
  hydra.sweep.dir=./outputs/HDM05

### Experiments on GyroBN under 3Block architeture
[ $? -eq 0 ] && python GyroBNGr.py -m\
  dataset.path=$path\
  nnet=GyroGr\
  nnet.model.architecture=[[93,93],[47,47],[24,24]]\
  nnet.model.is_pooling=True\
  nnet.model.is_final_pooling=False\
  nnet.BN_param.is_bn=True,False\
  nnet.BN_param.bn_type=GyroGrBN\
  hydra.job_logging.handlers.file.filename=3Block.log\
  hydra.run.dir=./outputs/HDM05\
  hydra.sweep.dir=./outputs/HDM05

### Experiments on GyroBN under 4Block architeture
[ $? -eq 0 ] && python GyroBNGr.py -m\
  dataset.path=$path\
  nnet=GyroGr\
  nnet.model.architecture=[[93,93],[47,47],[24,24],[12,12]]\
  nnet.model.is_pooling=True\
  nnet.model.is_final_pooling=False\
  nnet.BN_param.is_bn=True,False\
  nnet.BN_param.bn_type=GyroGrBN\
  hydra.job_logging.handlers.file.filename=4Block.log\
  hydra.run.dir=./outputs/HDM05\
  hydra.sweep.dir=./outputs/HDM05

