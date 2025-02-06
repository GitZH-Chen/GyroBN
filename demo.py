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
poincare_ball = Poincare()  # Assuming this handles the Poincaré ball manifold
random_hyperbolic_data = poincare_ball.projx(random_euclidean_vectors)  # Generate random points in the Poincaré ball

rbn_h = GyroBNH(dim=n,manifold=poincare_ball)  # Initialize GyroBNH for hyperbolic normalization
output_hyperbolic = rbn_h(random_hyperbolic_data)  # Apply GyroBNH

# Print shape to verify outputs
print("Grassmannian BN output shape:", output_grassmann.shape)
print("Hyperbolic BN output shape:", output_hyperbolic.shape)
