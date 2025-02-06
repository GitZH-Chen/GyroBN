import matplotlib.pyplot as plt
import torch as th
import random
import numpy as np
from Visualization.Grassmannian.utils_gras import visualize_gyrobn_gras, flip_z_if_negative
from Visualization.Poincare.utils_poincare import visualize_gyrobn_poincare, generate_random_poincare_vectors

# Step 1: Set random seeds for reproducibility
seed = 1024
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)
th.cuda.manual_seed(seed)

saved_name = 'GyroBN_vis_grass_poincare.png'
bs = 256
sample_size = 2
mean_size = 100
subplot_size = 5
font_size = 12  # Control font size here
mean_marker = 'o'
karcher_steps=1

# Step 2: Generate a batch of Grassmannians
matrices = np.random.uniform(low=-1000, high=1000, size=[bs, 3, 1])
norms = np.linalg.norm(matrices, axis=1, keepdims=True)
input_grassmannians = flip_z_if_negative(matrices / norms)

# Convert input_grassmannians to a torch tensor
input_grassmannians_torch = th.tensor(input_grassmannians, dtype=th.float64)

# Step 3: Generate a batch of Poincaré vectors (for hyperbolic space)
# input_poincare = generate_random_poincare_vectors(int(bs/2), 3)
input_poincare = generate_random_poincare_vectors(bs, 3)

# Convert input_poincare to a torch tensor
input_poincare_torch = th.tensor(input_poincare, dtype=th.float64)

# Example Usage with a 1x4 subplot grid:
fig, axs = plt.subplots(1, 4, subplot_kw={'projection': '3d'}, figsize=(subplot_size * 4, subplot_size + 1))

# Visualize GyroBN on the first two subplots (Grassmannian visualization)
visualize_gyrobn_gras([axs[0], axs[1]], input_grassmannians_torch, font_size=font_size, sample_size=sample_size,
                      mean_size=mean_size, mean_marker=mean_marker, karcher_steps=karcher_steps)

# Visualize GyroBN on the last two subplots (Poincaré ball visualization)
visualize_gyrobn_poincare([axs[2], axs[3]], input_poincare_torch, font_size=font_size, sample_size=sample_size,
                          mean_size=mean_size, mean_marker=mean_marker, karcher_steps=karcher_steps)

# Set specific shared titles for two different sections using fig.text
joint_title_fs = font_size + 4

# Shared title for the first two subplots (GyroBN visualizations on Grassmannians)
fig.text(0.25, 0.92, "GyroBN on the Grassmannian", ha='center', fontsize=joint_title_fs, weight='bold')

# Shared title for the third and fourth subplots (GyroBN visualizations on the Poincaré ball)
fig.text(0.75, 0.92, "GyroBN on the Poincaré Ball", ha='center', fontsize=joint_title_fs, weight='bold')

# Display the plot
plt.tight_layout(pad=3)
# plt.tight_layout(pad=3)
# plt.savefig(saved_name, dpi=400)  # Save figure with the provided name and resolution
plt.show()
