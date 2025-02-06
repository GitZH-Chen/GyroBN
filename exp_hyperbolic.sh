#!/bin/bash

# pubmed requires about 24932MiB GPU
path=/data #change this to your data folder

#--- HNN ---
[ $? -eq 0 ] && python GyroBNH.py -m\
  dataset.dataset=cora\
  dataset.path=$path\
  nnet.BN_param.is_bn=False\
  nnet.optimizer.weight_decay=0

[ $? -eq 0 ] && python GyroBNH.py -m\
  dataset.dataset=disease_lp,airport,pubmed\
  dataset.path=$path\
  nnet.BN_param.is_bn=False\
  nnet.optimizer.weight_decay=1e-3

#--- GyroBNH ---
[ $? -eq 0 ] && python GyroBNH.py -m\
  dataset.dataset=cora\
  dataset.path=$path\
  nnet.BN_param.is_bn=True\
  nnet.BN_param.bn_type=GyroBNH\
  nnet.optimizer.weight_decay=0

[ $? -eq 0 ] && python GyroBNH.py -m\
  dataset.dataset=disease_lp,airport,pubmed\
  dataset.path=$path\
  nnet.BN_param.is_bn=True\
  nnet.BN_param.bn_type=GyroBNH\
  nnet.optimizer.weight_decay=1e-3

#--- RBNH ---
[ $? -eq 0 ] && python GyroBNH.py -m\
  dataset.dataset=cora\
  dataset.path=$path\
  nnet.BN_param.is_bn=True\
  nnet.BN_param.bn_type=RBNH\
  nnet.optimizer.weight_decay=0

[ $? -eq 0 ] && python GyroBNH.py -m\
  dataset.dataset=disease_lp,airport,pubmed\
  dataset.path=$path\
  nnet.BN_param.is_bn=True\
  nnet.BN_param.bn_type=RBNH\
  nnet.optimizer.weight_decay=1e-3