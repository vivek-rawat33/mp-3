from numpy import *
from matplotlib.pyplot import *
#exp 7 
fn = lambda t: 3*sin(2*pi*t) + sin(8*pi*t) + 0.5*sin(14*pi*t)
gn= lambda t: exp(-t**2)
def dft(f, N, a, b):
    t = linspace(a, b, N)
    ft = array([f(i) for i in t])  # fixed
    w = complex(cos(2*pi/N), -sin(2*pi/N))
    
    wn = zeros((N, N), dtype=complex)
    for k in range(N):
        for i in range(N):
            wn[k, i] = w**(k * i)
    
    Fk = dot(wn, ft)
    return Fk, t, ft

def plot_dft(N, fn, a, b):
    Fk, x, fx = dft(fn, N, a, b)
    
    figure(figsize=(14,5))
    
    # Original function
    subplot(1,2,1)
    plot(x, fx, label="f(x)")
    xlabel("x")
    ylabel("f(x)")
    title("Function f(x)")
    grid(True)
    legend()
    
    # Magnitude spectrum
    subplot(1,2,2)
    k = range(N)
    # stem(k, [abs(val) for val in Fk], basefmt=" ", label="|F(k)|")
    stem(k, [abs(val) for val in Fk], basefmt=" ", markerfmt='.', linefmt='b-', label="|F(k)|")
    tight_layout()
    xlabel("k")
    ylabel("|F(k)|")
    title("DFT Magnitude Spectrum")
    grid(True)
    legend()
    
    show()

plot_dft(100, fn, 0, 1)

plot_dft(200,gn,0,1)

def compare_dft_fft(N, fn, a, b):
    # Manual DFT
    Fk_manual, x, fx = dft(fn, N, a, b)
    
    # NumPy FFT
    Fk_fft = np.fft.fft(fx)
    
    # Error check
    error = np.abs(Fk_manual - Fk_fft)
    max_error = np.max(error)
    
    print(f"Maximum absolute error between manual DFT and np.fft.fft: {max_error:.2e}")
    
    subplot(1,2,1)
    k = range(N)
    stem(k, abs(Fk_manual), basefmt=" ", markerfmt='.', linefmt='b-', label="|DFT (manual)|")
    xlabel("k")
    ylabel("|F(k)|")
    title("Manual DFT")
    grid(True)
    legend()

    subplot(1,2,2)
    k = range(N)
    stem(k, abs(Fk_fft), basefmt=" ", markerfmt='.', linefmt='b-', label="|FFT (NumPy)|")
    xlabel("k")
    ylabel("|F(k)|")
    title("using Built-in FFT")
    legend()
    grid(True)
    tight_layout()    
    show()

compare_dft_fft(100, fn, 0, 1)
compare_dft_fft(200,gn,0,1)