
from math import sqrt
import numpy as np
from matplotlib.pyplot import *

# Coefficients
pn = lambda x: 0
qn = lambda x: 0
rn = lambda x: -9.8

# Domain and boundary conditions
a, b = 0, 5
y0, y5 = 50, sqrt(10)

# Grid
n = int(input("enter n:"))             # number of sub-intervals
h = (b - a) / n
x = np.linspace(a, b, n+1)  # n+1 points including boundaries
x_ = np.linspace(a, b, 50)  # n+1 points including boundaries

# Matrix for interior points (size n-1)
A = np.zeros((n-1, n-1))
r = np.zeros(n-1)

# Fill RHS for interior points
for i in range(1, n):
    r[i-1] = h**2 * rn(x[i])

# Fill matrix A using finite difference coefficients
for i in range(n-1):
    xi = x[i+1]  # interior x
    if i > 0:
        A[i, i-1] = 1 - h/2 * pn(xi)
    A[i, i] = -2 + h**2 * qn(xi)
    if i < n-2:
        A[i, i+1] = 1 + h/2 * pn(xi)
# Apply boundary contributions
r[0] -= (1 - h/2 * pn(x[1])) * y0
r[-1] -= (1 + h/2 * pn(x[n-1])) * y5

# Solve system
y_interior = np.linalg.solve(A, r)

# Append boundary values
y_fdm = np.zeros(n+1)
y_fdm[0] = y0
y_fdm[1:n] = y_interior
y_fdm[n] = y5

# Analytic solution for comparison
y_exact = -4.9 * x_**2 + 15.1324555 * x_ + 50

# Plot
plot(x, y_fdm, label="Finite Difference Solution")
plot(x_, y_exact, label="Analytic Solution", linestyle='--')
xlabel("time(in seconds)")
ylabel("(displacement(in m))")
title("FDM vs Analytic Solution")
legend()
grid()
show()



#original plot
y0 = 50.0
t1 = 5.0
y1 = sqrt(10)

# compute g from y(5) = 50 - 0.5*g*5^2
g = 2*(y0 - y1) / (t1**2)

def y(t):
    return y0 - 0.5 * g * t**2

print("g =", g)
print("y(0) =", y(0))
print("y(5) =", y(5), " (target:", y1, ")")

# optional: plot
t = np.linspace(0, 6, 200)
plot(t, y(t))
scatter([0, 5], [y(0), y(5)], color="red")
xlabel("t")
ylabel("y(t)")
grid(True)
show()