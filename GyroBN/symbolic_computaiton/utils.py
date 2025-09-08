import sympy as sp
def compute_gyr_xy_coeffs(K, xy, xu, yu, X2, Y2):
    """Compute the coefficients of x and y in gyr[x, y](u)"""
    A = -K**2 * xu * Y2 - K * yu + 2 * K**2 * xy * yu
    B = -K**2 * yu * X2 + K * xu
    D = 1 - 2 * K * xy + K**2 * X2 * Y2
    coeff_x = sp.simplify(2 * A / D)
    coeff_y = sp.simplify(2 * B / D)
    return coeff_x, coeff_y