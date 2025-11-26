from math import sqrt
import numpy as np
from matplotlib.pyplot import *

# Coefficients for y'' = -9.8
pn = lambda x: 0
qn = lambda x: 0
rn = lambda x: -9.8

a, b = 0, 5
y0, y5 = 50, sqrt(10)

n = int(input("Enter n: "))
h = (b - a) / n
x = np.linspace(a, b, n + 1)

A = np.zeros((n - 1, n - 1))
r = np.zeros(n - 1)

for i in range(n - 1):
    xi = x[i + 1]
    if i > 0:
        A[i, i - 1] = 1 - h / 2 * pn(xi)
    A[i, i] = -2 + h ** 2 * qn(xi)
    if i < n - 2:
        A[i, i + 1] = 1 + h / 2 * pn(xi)
    r[i] = h ** 2 * rn(xi)

r[0] -= (1 - h / 2 * pn(x[1])) * y0
r[-1] -= (1 + h / 2 * pn(x[n - 1])) * y5

# Solve system
y_internal = np.linalg.solve(A, r)

y_fdm = np.zeros(n + 1)
y_fdm[0] = y0
y_fdm[1:n] = y_internal
y_fdm[n] = y5

x_ = np.linspace(a, b, 100)
y_analytic = -4.9 * x_**2 + 15.132455 * x_ + 50

plot(x, y_fdm, 'o-', label='Finite Difference Solution')
plot(x_, y_analytic, '--', label='Analytic Solution')
xlabel('time ( seconds )')
ylabel('displacement( in m )')
title("FDM vs Analytic solution")
grid(True)
legend()
show()
