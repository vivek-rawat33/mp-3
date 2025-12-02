from numpy import *
from cmath import *
from matplotlib.pyplot import *
import math as M
# exp 6
# Small DFT example
N = 4
f = [0,1,4,9]

def dft(N, f):
    w = complex(cos((2*pi)/N), -sin((2*pi)/N))
    Fk = []
    for k in range(N):
        s = 0
        for i in range(N):
            s += (w**(i*k)) * f[i]
        Fk.append(s)
    return Fk

fk = dft(N,f)
X_numpy = np.fft.fft(f)
print("Input Signal f[n]:", f)
print("\nManual DFT X[k]:")
for k, Xk in enumerate(fk):
    print(f"X[{k}] = {Xk:.2f}")

print("\nNumPy DFT X[k]:")
for k, Xk in enumerate(X_numpy):
    print(f"X[{k}] = {Xk:.2f}")
    
    
# DFT of function x*exp(-x)
fn = lambda x: x*M.exp(-x)
N = 128
a, b = 0, 8

def plot_dft(N, fn, a, b):
    dx = (b - a)/N
    x = [a + i*dx for i in range(N)]
    fx = [fn(xi) for xi in x]
    fk = dft(N,fx)
    figure(figsize=(10,6))
    subplot(1,2,1)
    plot(x, fx, label="f(x)")
    xlabel("x")
    ylabel("f(x)")
    title("Function f(x)")
    grid(True)
    legend()
    
    # Plot magnitude of DFT
    subplot(1,2,2)
    k = range(N)
    # plot(k, [abs(val) for val in fk], label="|F(k)|")
    stem(k, [abs(val) for val in fk], label="|F(k)|")
    xlabel("k")
    ylabel("|F(k)|")
    title("DFT Magnitude")
    grid(True)
    legend()
    show()

plot_dft(N, fn, a, b)
