import torch
import matplotlib.pyplot as plt

from GyroBN.Geometry.constantcurvature.utils.utils_ccs import sindiv

# Input tensor near zero, requires grad
x = torch.linspace(-1e-2, 1e-2, steps=500, requires_grad=True)

# Reference: sin(x) / x
y_ref = torch.sin(x) / x
grad_ref = torch.autograd.grad(y_ref.sum(), x, create_graph=True)[0]

y_stable = sindiv(x)
grad_stable = torch.autograd.grad(y_stable.sum(), x, create_graph=True)[0]

# Plot comparison
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x.detach(), y_ref.detach(), label='torch.sin(x)/x')
plt.plot(x.detach(), y_stable.detach(), '--', label='sindiv(x)')
plt.title("Forward: sin(x)/x vs sindiv(x)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x.detach(), grad_ref.detach(), label='grad torch.sin(x)/x')
plt.plot(x.detach(), grad_stable.detach(), '--', label='grad sindiv(x)')
plt.title("Backward: gradient comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
