#!/bin/bash

#---- SPDMLR and LieMLR on the HDM05_SPD ----

#---- SPDMLR ----
path=/data #change this to your data folder

### Ablations on GyroGr with different RBN under 1Block architeture
[ $? -eq 0 ] && python GyroBNGr.py -m\
  dataset.path=$path\
  nnet=GyroGr\
  nnet.model.architecture=[[93,93]]\
  nnet.model.is_pooling=True\
  nnet.model.is_final_pooling=True\
  nnet.BN_param.is_bn=True\
  nnet.BN_param.bn_type=GyroBNGr,RBNGr,ManifoldNormGr\
  hydra.job_logging.handlers.file.filename=1Block.log\
  hydra.run.dir=./outputs/HDM05\
  hydra.sweep.dir=./outputs/HDM05

