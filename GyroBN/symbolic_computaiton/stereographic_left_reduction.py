import sympy as sp

# === Goal ===
# Verify the gyrogroup left loop law in κ-geometry:
#   gyr[a, b](w) == gyr[a ⊕ b, b](w)
#
# Strategy:
# 1. Write both sides as linear combinations of a and b:
#       gyr[a, b](w) = w + A1 * a + B1 * b
#       gyr[a⊕b, b](w) = w + A2 * a + B2 * b
# 2. Express a⊕b = (A * a + B * b) / D
# 3. Substitute inner products and norms into symbolic expressions
# 4. Compare A1 == A2 and B1 == B2 structurally (symbolically)

# Define symbolic variables
K = sp.Symbol('K', real=True, positive=True)   # Curvature
ab, ac, bc = sp.symbols('ab ac bc')            # <a,b>, <a,c>, <b,c>
A2, B2 = sp.symbols('A2 B2')                   # ||a||^2, ||b||^2
aw = sp.Symbol('aw')                           # <a, w>
bw = sp.Symbol('bw')                           # <b, w>

def compute_gyr_ab_coeffs(K, ab, aw, bw, A2, B2):
    """
    Compute the coefficients of a and b in:
        gyr[a, b](w) = w + coeff_a * a + coeff_b * b
    using inner products and norms only.
    """
    A1 = -K**2 * aw * B2 - K * bw + 2 * K**2 * ab * bw
    B1 = -K**2 * bw * A2 + K * aw
    D1 = 1 - 2 * K * ab + K**2 * A2 * B2
    coeff_a = sp.simplify(2 * A1 / D1)
    coeff_b = sp.simplify(2 * B1 / D1)
    return coeff_a, coeff_b

# Step 1: Möbius addition — write a⊕b = (A*a + B*b)/D
A = 1 - 2*K*ab - K*B2
B = 1 + K*A2
D = 1 - 2*K*ab + K**2*A2*B2
A0_s = sp.simplify(A / D)
B0_s = sp.simplify(B / D)

# Step 2: gyr[a, b](w)
A1_s, B1_s = compute_gyr_ab_coeffs(K, ab, aw, bw, A2, B2)

# Step 3: Prepare inner products for gyr[a⊕b, b](w)
norm2_aplusb = (A**2 * A2 + 2*A*B*ab + B**2 * B2) / D**2     # ||a⊕b||^2
dot_aplusb_b = (A * ab + B * B2) / D                         # <a⊕b, b>
dot_aplusb_w = (A * aw + B * bw) / D                         # <a⊕b, w>

# Step 4: gyr[a⊕b, b](w)
A2_s, B2_s = compute_gyr_ab_coeffs(K, dot_aplusb_b, dot_aplusb_w, bw, norm2_aplusb, B2)

# Step 5: Expand gyr[a⊕b, b](w) back to a and b
A_rhs = sp.simplify(A2_s * A0_s)
B_rhs = sp.simplify(A2_s * B0_s + B2_s)

# Step 6: Compare with gyr[a, b](w)
A_equal = sp.simplify(A1_s - A_rhs) == 0
B_equal = sp.simplify(B1_s - B_rhs) == 0

# Output results
print("A_equal =", A_equal)  # Should be True
print("B_equal =", B_equal)  # Should be True
