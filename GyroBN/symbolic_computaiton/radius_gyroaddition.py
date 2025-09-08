# SymPy checker for the heavy algebra in Step 3.
# Verifies:  ||w||^2 = N/D
import sympy as sp

# Symbols (assume all are real; a, b, K are nonzero to avoid spurious zero denominators)
K, a, b = sp.symbols('K a b', nonzero=True)
nx, ny, s = sp.symbols('n_x n_y s_{xy}', real=True)

# Stereographic preimages: u = x_s / a, v = y_s / b
# Their norms and inner product in terms of (a, b, nx, ny, s)
u2 = nx / a**2
v2 = ny / b**2
uv = s / (a*b)

# Coefficients in the Möbius addition formula
A = 1 - 2*K*uv - K*v2
B = 1 + K*u2
Delta = 1 - 2*K*uv + K**2*u2*v2

# Squared norm of w = (A u + B v) / Delta
w2 = (A**2*u2 + B**2*v2 + 2*A*B*uv) / Delta**2

# Targets
D = a**2*b**2 - 2*K*a*b*s + K**2*nx*ny
N = a**2*ny + 2*a*b*s + b**2*nx

# Robust check: take the difference, put over a common denominator, and test the numerator
diff = sp.together(w2 - N/D)
num, den = sp.together(diff).as_numer_denom()
ok_w2 = sp.simplify(sp.factor(num)) == 0

print("||w||^2 == N/D ? ->", ok_w2)
