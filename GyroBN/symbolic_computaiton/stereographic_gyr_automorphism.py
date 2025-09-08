"""
This script symbolically verifies the automorphism property of gyration
in the stereographic gyrovector space model with constant curvature K > 0.

We aim to verify the identity:
    gyr[x, y](w ⊕ z) = gyr[x, y](w) ⊕ gyr[x, y](z)

This is done by:
1. Symbolically expanding the right-hand side (RHS):
       gyr[x, y](w) ⊕ gyr[x, y](z),
   as a linear combination of x, y, w, z with coefficients in terms of
   inner products and norms.
2. Separately computing the left-hand side (LHS):
       gyr[x, y](w ⊕ z),
   using the Möbius addition formula and the symbolic structure of
   the gyration map.
3. Comparing the coefficients of x, y, w, z on both sides to ensure
   symbolic equivalence via simplification.

All variables like ⟨x,w⟩, ⟨y,z⟩, ∥x∥² etc. are abstract symbols, allowing verification in arbitrary dimension.
"""
import sympy as sp

from utils import compute_gyr_xy_coeffs

K = sp.Symbol('K', real=True, positive=True)
xw, xz, yw, yz, wz = sp.symbols('xw xz yw yz wz')  # inner products
xy = sp.Symbol('xy')                               # <x, y>
X2, Y2 = sp.symbols('X2 Y2')                        # ||x||^2, ||y||^2
W2, Z2 = sp.symbols('W2 Z2')                        # ||w||^2, ||z||^2

# Step 1: Compute gyr[x,y](w) = w + f1*x + g1*y
f1, g1 = compute_gyr_xy_coeffs(K, xy, xw, yw, X2, Y2)
# Step 2: Compute gyr[x,y](z) = z + f2*x + g2*y
f2, g2 = compute_gyr_xy_coeffs(K, xy, xz, yz, X2, Y2)

# Step 3: Compute inner products <gyr[w], gyr[z]> and norms
# < gyr[x, y](w), gyr[x, y](z) > =< w + f_1 x + g_1 y, z + f_2 x + g_2 y >
dot_gw_gz = wz + f1*xz + f2*xw + g1*yz + g2*yw + f1*f2*X2 + g1*g2*Y2 + (f1*g2 + f2*g1)*xy
# < gyr[x, y](w), gyr[x, y](w) > =< w + f_1 x + g_1 y, w + f_1 x + g_1 y >
norm_gw_sq = W2 + 2*f1*xw + 2*g1*yw + f1**2*X2 + 2*f1*g1*xy + g1**2*Y2
# < gyr[x, y](z), gyr[x, y](z) > =< z + f_2 x + g_2 y, z + f_2 x + g_2 y >
norm_gz_sq = Z2 + 2*f2*xz + 2*g2*yz + f2**2*X2 + 2*f2*g2*xy + g2**2*Y2

# Step 4: Möbius addition: gyr[x,y](w) ⊕ gyr[x,y(z)
A_top = 1 - 2*K*dot_gw_gz - K*norm_gz_sq
B_top = 1 + K*norm_gw_sq
D_bot = 1 - 2*K*dot_gw_gz + K**2 * norm_gw_sq * norm_gz_sq

# Step 5: Compute RHS coefficients of x, y, w, z
coeffs_rhs = {
    "x": sp.simplify(A_top * f1 + B_top * f2) / D_bot,
    "y": sp.simplify(A_top * g1 + B_top * g2) / D_bot,
    "w": sp.simplify(A_top / D_bot),
    "z": sp.simplify(B_top / D_bot),
}

# Step 6: Compute LHS: gyr[x,y](w ⊕ z)
A_wz = 1 - 2*K*wz - K*Z2
B_wz = 1 + K*W2
D_wz = 1 - 2*K*wz + K**2 * W2 * Z2
a0 = sp.simplify(A_wz / D_wz)
b0 = sp.simplify(B_wz / D_wz)

# Inner products for gyr[x,y](w⊕z)
x_wz = a0 * xw + b0 * xz
y_wz = a0 * yw + b0 * yz
norm_wz_sq = a0**2 * W2 + 2 * a0 * b0 * wz + b0**2 * Z2

# Apply gyr[x,y](w⊕z)
f_x, f_y = compute_gyr_xy_coeffs(K, xy, x_wz, y_wz, X2, Y2)

# Step 7: Compute LHS coefficients of x, y, w, z
coeffs_lhs = {
    "x": sp.simplify(f_x),
    "y": sp.simplify(f_y),
    "w": sp.simplify(a0),
    "z": sp.simplify(b0),
}

# Step 8: Verify equality
equalities = {
    key: sp.simplify(coeffs_rhs[key] - coeffs_lhs[key]) == 0
    for key in coeffs_rhs
}

print("equalities =", equalities)  # Should be True

