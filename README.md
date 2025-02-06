[//]: # ([<img src="https://img.shields.io/badge/arXiv-2409.19433-b31b1b"></img>]&#40;https://arxiv.org/abs/2409.19433&#41;)
[<img src="https://img.shields.io/badge/OpenReview|forum-d1NWq4PjJW-8c1b13"></img>](https://openreview.net/forum?id=d1NWq4PjJW)
[<img src="https://img.shields.io/badge/OpenReview|pdf-d1NWq4PjJW-8c1b13"></img>](https://openreview.net/pdf?id=d1NWq4PjJW)

# Gyrogroup Batch Normalization

This is the official code for our ICLR 2025 publication: *Gyrogroup Batch Normalization*. 

If you find this project helpful, please consider citing us as follows:

```bib
@inproceedings{chen2025gyrogroup,
    title={Gyrogroup Batch Normalization},
    author={Ziheng Chen and Yue Song and Xiaojun Wu and Nicu Sebe},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025}
}
```
If you have any problem, do not hesitate to contact me via ziheng_ch@163.com.

## Implementations
This source code contains GyroBN on the Grassmannian and hyperbolic spaces. 
Besides, we also implement RBN and ManifoldNorm on the Grassmannian, as shown in Tab. 3.

## Requirements

Install necessary dependencies by `conda`:

```setup
conda env create --file environment.yaml
```

**Note** that the [hydra](https://hydra.cc/) package is used to manage configuration files.

## Demos
 Demos of Grassmannian and Hyperbolic GyroBN, which can be found in `demo.py`:

```python
import torch as th
from RieNets.grnets.GrBN import GyroGrBN
from Geometry.Grassmannian import GrassmannianGyro
from RieNets.hnns.layers.GyroBNH import GyroBNH
from frechetmean import Poincare

# --- Typical use of Grassmannian GyroBN
bs, c, n, p = 32, 8, 30, 10
grassmannian = GrassmannianGyro(n=n, p=p)
random_data = grassmannian.random(bs, c, n, p)
rbn = GyroGrBN(shape=[c, n, p])
output_grassmann = rbn(random_data)

# --- Typical use of GyroBNH in the Poincaré ball
random_euclidean_vectors = th.randn(bs, n)/10
poincare_ball = Poincare() 
random_hyperbolic_data = poincare_ball.projx(random_euclidean_vectors)  # Generate random points in the Poincaré ball

rbn_h = GyroBNH(dim=n,manifold=poincare_ball)  # Initialize GyroBNH for hyperbolic normalization
output_hyperbolic = rbn_h(random_hyperbolic_data)  # Apply GyroBNH

# Print shape to verify outputs
print("Grassmannian BN output shape:", output_grassmann.shape)
print("Hyperbolic BN output shape:", output_hyperbolic.shape)

```

### Dataset

The preprocessed Grassmannian from the HDM05 dataset can be found [here](https://www.dropbox.com/scl/fi/chzvrg3srq6jwntqlr8n4/HDM05_Grassmannian.zip?rlkey=r4b87nybepv58bu8re14jp57d&st=vyy888lc&dl=0). 
The link prediction datasets can be found [here](https://www.dropbox.com/scl/fi/5rm3lwp367spd58xp1vil/Graph-LP.zip?rlkey=t32yiestqmqfdcuodqax8n4ny&st=lqbd8x3m&dl=0).
Please download the datasets and put them in your folder.
If necessary, change the `path` in `conf/dataset/HDM05.yaml` and `conf/dataset/LP.yaml`,

### Running experiments
#### Grassmannian
Please run this command for the main experiments on the HDM05 (Tab. 3):
```train
bash exp_grassmannian.sh
```
Please run this command for the KBlock ablations on the HDM05 (Tab. 4):
```train
bash exp_grassmannian.sh
```
The tensorboard results on the HDM05 under 1-block architecture:

Acc on the HDM05 dataset.
![GyroBNGr](Acc-GyroGr-HDM05.png)

**Note:** You can change the `path` in `exp_xxx.sh`, which will override the hydra config.

#### Hyperbolic
Please run this command for HNN with or without GyroBN-H or RBN-H on the link prediction task. (Tab. 5):
```train
bash exp_hyperbolic.sh
```
ROC on three typical link prediction datasets.
![GyroBNH](Acc-GyroBNH-LP.png)

### Visualization

Visualization code can be found in `./Visualization/gyrobn_gras_hyperbolic.py`, corresponding to Fig. 1.



