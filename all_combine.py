#complex number 
from cmath import *
import matplotlib.pyplot as plt 
import numpy as np
import math as M
z1 = 2+3j
z2 = 2-5j
z3 = -3 -1j 

print(f"addition is {z1+z2}")
print(f"subtraction is {z1-z2}")
print(f"multiplication is {z1*z2}")
print(f"division is {z1/z2}")

def argument(z):
    x, y = z.real, z.imag
    if x == 0 and y == 0:
        print("Argument undefined")
        return
    if x == 0:
        arg = M.pi/2 if y > 0 else -M.pi/2

    elif y == 0:
        arg = 0 if x > 0 else M.pi

    # All quadrants
    else:
        arg = M.atan(y/x)
        if x < 0:                      # 2nd or 3rd quadrant
            arg += M.pi if y > 0 else -M.pi

    print(f"argument of the complex {z} is {arg}")
argument(z3)
print(f"difference between inbuilt and function defined {abs(argument(z3) - phase(z3))}")
        
def modules(z):
    x,y = z.real , z.imag
    return M.sqrt(x**2 - y**2)

print(f"Magnitude of complex {z3} is {modules(z3)}")

###########################################################################################################


#nth root of complex number 
def nth_root(z,n):
    r = abs(z)
    roots = []
    for i in range(n):
        theta = (phase(z)+ 2*M.pi *i )/n
        root = (r**(1/n))*exp(theta*1j)
        roots.append(root)
    return roots 
roots = nth_root(z1,3)
print(f"roots are {roots}")

for root in roots:
    plt.plot([0,root.real],[0,root.imag] ,label=f"root {root}")
plt.title(f"roots of complex {z1}")
plt.grid(True)
plt.legend()
plt.xlabel("Real axis")
plt.ylabel("imaginary axis")
plt.show()


##############################################################################################################
#f(z) = z^2 function plot 
x1 = np.linspace(-10,10,100)
x2 = np.linspace(-1,2,100)
y1 = np.linspace(-10,10,100)
y2= np.linspace(-2,3,100)

z1 = [complex(x1[i],y1[i]) for i in range(100)]
z2 = [complex(x2[i],y2[i]) for i in range(100)]

def f(x,y):
    n= len(x)
    f = []
    for i in range(n):
        u= x[i]**2 - y[i]**2
        v = 2*x[i]*y[i]
        f.append(complex(u,v))
    return f

fz1 = f(x1,y1)
fz2 = f(x2,y2)

def plotting(fz,z):
    fz_real = [fz.real for fz in fz]
    fz_imag = [fz.imag for fz in fz]
    z_real = [z.real for z in z]
    z_imag = [z.imag for z in z]
    plt.plot(fz_real,fz_imag, label="fz", color="red")
    plt.plot(z_real,z_imag,label="z",color='blue')
    plt.scatter([z_real[0],z_real[-1]],[z_imag[0],z_imag[-1]] ,color='black')
    for i in range(len(z)):
        plt.plot([z_real[i],fz_real[i]],[z_imag[i],fz_imag[i]],linestyle="--", color="grey")
        plt.grid(True)
        plt.legend()
    plt.show()
plotting(fz1,z1)
plotting(fz2,z2)


##################################################################################################################


#dft and fft 
N= 4
f =[0,1,4,9]
def dft_transform(f):
    N = len(f)
    F=[]
    for k in range(N):
        s = 0
        for n in range(N):
            angle = - 2*M.pi*k*n/N
            s+= f[n]*exp(1j*angle)
        F.append(s)
    return F
dft_signal = dft_transform(f)
for i,val in enumerate(dft_signal):
    print(f"signal {i+1} is {val}")
    
fft_signal = np.fft.fft(f)
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.grid(True)
plt.stem([1,2,3,4],np.abs(dft_signal))
plt.title("DFT Magnitude signal")
plt.subplot(1,2,2)
plt.stem([1,2,3,4],np.abs(fft_signal))
plt.title("FFT Magnitude signals")
plt.grid(True)
plt.show()


# f(x) = xe^-x
fn = lambda x: x*M.exp(-x)
N=128
a,b = 0,8
x = np.linspace(a,b,N)
fx = [fn(x) for x in x]
dft_signal = dft_transform(fx)
k = range(N)
# using fft inbuilt function
fft_signal = np.fft.fft(fx)
def dft_plotting(k,dft_signal,x,fx,title):
    plt.figure(figsize=(11,6))
    plt.subplot(1,2,1)
    plt.plot(x,fx,label="Original signal")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.stem(k , [abs(dft) for dft in dft_signal ],label="DFT Magnitude signal")
    plt.title(f"{title}")
    plt.legend()
    plt.grid(True)
    plt.xlabel("k")
    plt.ylabel("|F(k)|")
    plt.show()
dft_plotting(k,dft_signal,x,fx,"DFT manual")
dft_plotting(k,fft_signal,x,fx,"FFT inbuilt")


# f = 3sin(2pit) + sin(8pit) + 0.5sin(14pit)
fn = lambda t: 3*sin(2*pi*t) + sin(8*pi*t) + 0.5*sin(14*pi*t)
a,b = 0,1 
N =100
x = np.linspace(a,b,N)
fx = [fn(x) for x in x]
dft_signal1 = dft_transform(fx)
k1 = range(N)
fft_signal1 = np.fft.fft(fx)
dft_plotting(k1,dft_signal1,x,fx,"Manual")
dft_plotting(k1,fft_signal1,x,fx,"inbuilt")


#g = e^(-t^2)
gn= lambda t: exp(-t**2)
a,b = 0,1
N = 200 
t = np.linspace(a,b,N)
gt = [ gn(t) for t in t ]
dft_signal2 = dft_transform(gt)
k2 = range(N)
fft_signal2 = np.fft.fft(gt)
dft_plotting(k2,dft_signal2,t,gt,"Manual")
dft_plotting(k2,fft_signal2,t,gt,"inbuilt")

def compare_fft(k,fft_signal,dft_signal):
    error = np.abs(fft_signal-dft_signal)
    max_error = np.max(error)
    plt.figure(figsize=(11,7))
    plt.subplot(1,2,1)
    plt.stem(k,[np.abs(dft) for dft in dft_signal],label = "DFT Manual plot")
    plt.xlabel("K")
    plt.ylabel("|F(K)|")
    plt.grid(True)
    plt.legend()
    plt.subplot(1,2,2)
    plt.stem(k,[np.abs(fft) for fft in fft_signal],label = "FFT Inbuilt plot")
    plt.xlabel("K")
    plt.ylabel("|F(K)|")
    plt.grid(True)
    plt.legend()
    plt.suptitle("Comaprison between DFT and FFT transformation")
    plt.show()
    return max_error
error1 = compare_fft(k1,fft_signal1,dft_signal1)
error2 = compare_fft(k2,fft_signal2,dft_signal2)
print(f"error between dft and fft is {error1} of function f=3sin(2pit) + sin(8pit) + 0.5sin(14pit) \nerror between dft and fft is {error2} of function f = e^(-t^2)")
    