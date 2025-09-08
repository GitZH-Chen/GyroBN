import sympy as sp

# -----------------------------------------------------------------------------
# Symbolic verification of Axiom V2 in the stereographic model (K > 0):
# We check whether scalar multiplication satisfies:
#     (s + t) ⊗ x == (s ⊗ x) ⊕ (t ⊗ x)
# by comparing the coefficients along x in both sides.
# -----------------------------------------------------------------------------

# Define symbols
s, t, K, theta = sp.symbols('s t K theta', real=True, positive=True)
X2 = sp.Symbol('X2', real=True, positive=True)  # ||x||^2

# Define scalar multiplications
coeff_s = sp.tan(s * theta) / (sp.sqrt(K) * sp.sqrt(X2))
coeff_t = sp.tan(t * theta) / (sp.sqrt(K) * sp.sqrt(X2))
coeff_st = sp.tan((s + t) * theta) / (sp.sqrt(K) * sp.sqrt(X2))

# Let u = coeff_s * x, v = coeff_t * x
# Compute inner products and norms
uv = coeff_s * coeff_t * X2
u2 = coeff_s**2 * X2
v2 = coeff_t**2 * X2

# Möbius addition formula for K > 0
A = 1 - 2*K*uv - K*v2
B = 1 + K*u2
D = 1 - 2*K*uv + K**2 * u2 * v2
coeff_rhs = sp.simplify((A * coeff_s + B * coeff_t) / D)

# LHS: scalar multiplication of (s + t)
coeff_lhs = sp.simplify(coeff_st)

# Output
print("LHS coefficient (scalar on x):")
sp.pprint(coeff_lhs)
print("\nRHS coefficient (scalar on x):")
sp.pprint(coeff_rhs)
