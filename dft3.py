import numpy as np

f = np.array([0, 1, 4, 9])
N = len(f)
X_manual = []

for k in range(N):
    X_k = 0
    for n in range(N):
        X_k += f[n] * np.exp(-2j * np.pi * k * n / N)
    X_manual.append(X_k)

X_manual = np.array(X_manual)
X_numpy = np.fft.fft(f)

print("Input Signal f[n]:", f)
print("\nManual DFT X[k]:")
for k, Xk in enumerate(X_manual):
    print(f"X[{k}] = {Xk:.2f}")

print("\nNumPy DFT X[k]:")
for k, Xk in enumerate(X_numpy):
    print(f"X[{k}] = {Xk:.2f}")

