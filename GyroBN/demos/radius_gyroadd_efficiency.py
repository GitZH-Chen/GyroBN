import time
import torch
from GyroBN.Geometry import Sphere, Hyperboloid

# ============================================================
#      GyroBN Geometry - Sphere & Hyperboloid: Speed Test
# ============================================================
#
# Goal:
#   Compare runtime performance of:
#       - gyroadd (closed-form)
#       - _gyroadd_via_riem (composed Riemannian ops)
#   for both:
#       - Sphere (K>0)
#       - Hyperboloid (K<0)
#
# Output:
#   - Numerical correctness statistics (max/mean deviation)
#   - Speed (mean runtime per call over N repeats)
# ============================================================

def _to_device_dtype(x, device, dtype):
    """Ensure tensor is on the correct device and dtype."""
    return x.to(dtype=dtype, device=device)

def _sync(device):
    """Synchronize device to ensure accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()

@torch.no_grad()
def _bench_once(fn, *args, device=None, repeat=100):
    """Run `fn(*args)` `repeat` times and return total runtime in seconds."""
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = fn(*args)
    _sync(device)
    return time.perf_counter() - t0

@torch.no_grad()
def _speed_report(geom_name, K, x, y, geom, warmup=10, repeat=100, device=None, threshold=1e-6):
    # ---------- Correctness check ----------
    z_closed = geom.gyroadd(x, y)
    z_riem   = geom._gyroadd_via_riem(x, y)
    diff = (z_closed - z_riem).norm(dim=-1)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100
    ok = "✅" if max_diff < threshold else "❌"

    # ---------- Warm-up ----------
    for _ in range(warmup):
        _ = geom.gyroadd(x, y)
        _ = geom._gyroadd_via_riem(x, y)
    _sync(device)

    # ---------- Timing ----------
    t_closed = _bench_once(geom.gyroadd, x, y, device=device, repeat=repeat)
    t_riem   = _bench_once(geom._gyroadd_via_riem, x, y, device=device, repeat=repeat)

    avg_us_closed = (t_closed / repeat) * 1e6
    avg_us_riem   = (t_riem   / repeat) * 1e6
    speedup = avg_us_closed / avg_us_riem

    # ---------- Report ----------
    print(f"\n[{geom_name} (K={K:+.2f}) gyroadd Speed & Correctness Test]")
    print("------------------------------------------------------------")
    print(f"{ok} Correctness:")
    print(f"    Max ‖closed - via_riem‖ = {max_diff:.3e}")
    print(f"    Mean‖closed - via_riem‖ = {mean_diff:.3e}")
    print(f"    % within {threshold:.0e}      = {pass_ratio:.2f}%")
    print("------------------------------------------------------------")
    print(f"Speed (per call, mean over {repeat} repeats):")
    print(f"    gyroadd (closed-form):      {avg_us_closed:10.2f} μs")
    print(f"    _gyroadd_via_riem (ops):    {avg_us_riem:10.2f} μs")
    print("------------------------------------------------------------")
    print(f"Speedup (closed / via_riem )  ≈  {speedup*100:.4f}%")
    print("------------------------------------------------------------\n")

@torch.no_grad()
def run_sphere_gyroadd_speed_test(
    K=+1.0, batch_size=10000, dim=128, scale=None,
    warmup=10, repeat=100, device=None, dtype=torch.float64, threshold=1e-6
):
    """Sphere test (K>0)."""
    assert K > 0, "Sphere test requires K > 0"
    if scale is None:
        scale = 1.0 / dim

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    geom = Sphere(K=K).to(dtype)

    # Sample points on the manifold
    x = geom.random_normal(batch_size, dim, scale=scale)
    y = geom.random_normal(batch_size, dim, scale=scale)
    x = _to_device_dtype(x, device, dtype)
    y = _to_device_dtype(y, device, dtype)

    # Validate points
    ok_x, reason_x = geom._check_point_on_manifold(x)
    ok_y, reason_y = geom._check_point_on_manifold(y)
    if not ok_x: raise ValueError(f"[Sphere] x not on manifold: {reason_x}")
    if not ok_y: raise ValueError(f"[Sphere] y not on manifold: {reason_y}")

    _speed_report("Sphere", K, x, y, geom, warmup, repeat, device, threshold)

@torch.no_grad()
def run_hyperboloid_gyroadd_speed_test(
    K=-1.0, batch_size=10000, dim=128, scale=None,
    warmup=10, repeat=100, device=None, dtype=torch.float64, threshold=1e-6
):
    """Hyperboloid test (K<0)."""
    assert K < 0, "Hyperboloid test requires K < 0"
    if scale is None:
        scale = 1.0 / dim

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    geom = Hyperboloid(K=K).to(dtype)

    # Sample points on the manifold
    x = geom.random_normal(batch_size, dim, scale=scale)
    y = geom.random_normal(batch_size, dim, scale=scale)
    x = _to_device_dtype(x, device, dtype)
    y = _to_device_dtype(y, device, dtype)

    # Validate points
    ok_x, reason_x = geom._check_point_on_manifold(x)
    ok_y, reason_y = geom._check_point_on_manifold(y)
    if not ok_x: raise ValueError(f"[Hyperboloid] x not on manifold: {reason_x}")
    if not ok_y: raise ValueError(f"[Hyperboloid] y not on manifold: {reason_y}")

    _speed_report("Hyperboloid", K, x, y, geom, warmup, repeat, device, threshold)

# ============================================================
#                       Run Benchmark
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size=10000
    dim=128
    run_sphere_gyroadd_speed_test(K=+1.0,batch_size=batch_size,dim=dim)
    run_hyperboloid_gyroadd_speed_test(K=-1.0,batch_size=batch_size,dim=dim)
