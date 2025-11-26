import numpy as np
import matplotlib.pyplot as plt

n = 128
x = np.linspace(0, 8, n) 
f_x = x * np.exp(-x)

# Sampling interval (spacing between x values)
d = x[1] - x[0]

# Compute FFT and corresponding frequencies
X = np.fft.fft(f_x)
freqs = np.fft.fftfreq(n, d=d)

# Plot
plt.figure(figsize=(15, 6))

# Plot original signal
plt.subplot(1, 2, 1)
plt.plot(x, f_x)
plt.title("Original Signal f(x) = x·e^(-x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)

# Plot full DFT magnitude spectrum with frequency axis
plt.subplot(1, 2, 2)
plt.stem(freqs, np.abs(X), basefmt=" ")
plt.title("DFT Magnitude Spectrum |X[k]|")
plt.xlabel("Frequency (1/x units)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()
