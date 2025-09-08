import numpy as np
import torch as th
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from GyroBN.Geometry import *

from utils.utils_gras import flip_z_if_negative,visualize_gyrobn_gras
from utils.utils_ccs import visualize_gyrobn_ccs
from utils.utils_common import set_axis
from utils.utils_hyperboloid import rand_hyperboloid
from utils.utils_cor import plot_bn_cor,random_cor,generate_correlation_boundary

def set_seed(seed):
    seed = seed
    random.seed(seed)
    # th.cuda.set_device(args.gpu)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

# Set up the parameters
saved_name='GyroBN_illustration.png'
set_seed(1024)
bs = 400
sample_size = 2
mean_size = 100
subplot_size = 5
font_size = 12  # Control font size here
num_boundary=40
mean_marker='o'

fig = plt.figure(figsize=(subplot_size * 4, subplot_size * 4 + 1))
gs = gridspec.GridSpec(4, 4, figure=fig)

# 前三行保持 4×4
axs = np.empty((4, 4), dtype=object)
for i in range(3):        # row 0,1,2
    for j in range(4):
        axs[i,j] = fig.add_subplot(gs[i,j], projection='3d')

# 最后一行 (row=3)，合并为 1×2，但放在中间
axs[3,0] = fig.add_subplot(gs[3,1:2], projection='3d')   # 第2格
axs[3,1] = fig.add_subplot(gs[3,2:3], projection='3d')   # 第3格

#=== Grassmannians ===
matrices = np.random.uniform(low=-1000, high=1000, size=[int(bs/2), 3, 1])
norms = np.linalg.norm(matrices, axis=1, keepdims=True)
input_grassmannians = th.tensor(flip_z_if_negative(matrices / norms), dtype=th.float64)
visualize_gyrobn_gras([axs[0,0], axs[0,1]], input_grassmannians, font_size=font_size, sample_size=sample_size,
                      mean_size=mean_size, mean_marker=mean_marker)

#=== Poincare ===
pball = Stereographic(K=-1.0)
input_pball= pball.projx(th.randn(int(bs), 3)+1).to(th.float64)
visualize_gyrobn_ccs([axs[0,2], axs[0,3]], input_pball,
                          model='Stereographic', K=-1.,
                          font_size=font_size, sample_size=sample_size,
                          mean_size=mean_size, mean_marker=mean_marker)

#=== Hyperboloid ===
manifold = Hyperboloid(K=-1.0)
input_hyp = rand_hyperboloid(bs=bs, spatial_max=1.7)
visualize_gyrobn_ccs([axs[1,0], axs[1,1]], input_hyp,
                          model='Hyperboloid', K=-1.,
                          font_size=font_size, sample_size=sample_size,
                          mean_size=mean_size, mean_marker=mean_marker)
set_axis([axs[1,0], axs[1,1]],
         xset=[-2, 2, 1.],
         yset=[-2, 2, 1.],
         zset=[1, 3, 1])

#=== Klein ===
manifold = Klein(K=-1.0)
# input=manifold.projx(th.randn(int(bs*2), 3)+1).to(th.float64)
visualize_gyrobn_ccs([axs[1,2], axs[1,3]], input_pball,
                          model='Klein', K=-1.,
                          font_size=font_size, sample_size=sample_size,
                          mean_size=mean_size, mean_marker=mean_marker)

#=== Sphere ===
manifold = Sphere(K=1.0)
input=manifold.projx(th.randn(int(bs), 3)).to(th.float64)
visualize_gyrobn_ccs([axs[2,0], axs[2,1]], input,
                          model='Sphere', K=1.,
                          font_size=font_size, sample_size=sample_size,
                          mean_size=mean_size, mean_marker=mean_marker)
set_axis([axs[2,0], axs[2,1]],
         xset=[-1, 1, 0.5],
         yset=[-1, 1, 0.5],
         zset=[-1, 1, 0.5])

#=== Projected hypersphere ===
manifold = Stereographic(K=1.0)
input = (th.randn(int(bs), 3, dtype=th.float64) + th.tensor([0,0,0.8])).clamp(-2, 2)
visualize_gyrobn_ccs([axs[2,2], axs[2,3]], input,
                          model='Stereographic', K=1.,
                          font_size=font_size, sample_size=sample_size,
                          mean_size=mean_size, mean_marker=mean_marker)
set_axis([axs[2,2], axs[2,3]],
         xset=[-2, 2, 1],
         yset=[-2, 2, 1],
         zset=[-2, 2, 1])

# Correlation
cor_tensors = random_cor(bs,3,3).to(th.double)
cor_boundary = generate_correlation_boundary(num_matrices_per_case=bs)
plot_bn_cor(axs=(axs[3,0], axs[3,1]), input=cor_tensors, boundary=cor_boundary,max_bound=1,
            font_size=font_size, sample_size=sample_size, mean_size=mean_size,mean_marker=mean_marker)

# Add joint titles
joint_title_fs = font_size + 4

# Row 0
fig.text(0.25, 0.975, "GyroBN on the Grassmannian", ha='center',
         fontsize=joint_title_fs, weight='bold')
fig.text(0.75, 0.975, "GyroBN on the Poincaré Ball", ha='center',
         fontsize=joint_title_fs, weight='bold')

# Row 1
fig.text(0.25, 0.73, "GyroBN on the Hyperboloid", ha='center',
         fontsize=joint_title_fs, weight='bold')
fig.text(0.75, 0.73, "GyroBN on the Beltrami–Klein", ha='center',
         fontsize=joint_title_fs, weight='bold')

# Row 2
fig.text(0.25, 0.49, "GyroBN on the Sphere", ha='center',
         fontsize=joint_title_fs, weight='bold')
fig.text(0.75, 0.49, "GyroBN on the Projected Hypersphere", ha='center',
         fontsize=joint_title_fs, weight='bold')

# Row 3 (last row, centered)
fig.text(0.50, 0.245, "GyroBN on the Correlation", ha='center',
         fontsize=joint_title_fs, weight='bold')

# Adjust the space between rows
plt.subplots_adjust(hspace=0.2)  # Adjust this value as needed to increase/decrease space
plt.tight_layout(pad=3)
plt.savefig(saved_name, dpi=150)  # Save figure with the provided name and resolution
plt.show()  # Display the plot
