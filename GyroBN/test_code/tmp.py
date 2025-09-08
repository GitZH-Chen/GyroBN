import torch as th
from GyroBN.Geometry import Stereographic

def test_poincare_norms(dim=325, K=-1.0, std_list=[1e-3, 0.01, 0.1, 0.5, 1.0, 2.0], n_samples=1000, seed=42):
    th.manual_seed(seed)
    pball = Stereographic(K=K)   # Poincaré ball manifold (K<0)

    for std in std_list:
        # 在切空间 0 点采样高斯向量，然后映射到球上
        x = pball.random_normal(n_samples, dim, std=std)
        norms = th.norm(x, dim=-1)   # Euclidean norm in R^d
        print(f"\n=== std={std} ===")
        print(f"mean norm = {norms.mean().item():.4f}")
        print(f"max  norm = {norms.max().item():.4f}")
        print(f"95th pct   = {norms.kthvalue(int(0.95*n_samples)).values.item():.4f}")
        print(f"99th pct   = {norms.kthvalue(int(0.99*n_samples)).values.item():.4f}")

if __name__ == "__main__":
    test_poincare_norms()
