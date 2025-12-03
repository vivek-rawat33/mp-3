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
    return arg
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
    plt.plot([0,root.real],[0,root.imag] ,label=f"root {complex(round(root.real,3),round(root.imag,3))}")
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
# dft_plotting(k,dft_signal,x,fx,"DFT manual")
# dft_plotting(k,fft_signal,x,fx,"FFT inbuilt")


# f = 3sin(2pit) + sin(8pit) + 0.5sin(14pit)
fn = lambda t: 3*sin(2*pi*t) + sin(8*pi*t) + 0.5*sin(14*pi*t)
a,b = 0,1 
N =100
x = np.linspace(a,b,N)
fx = [fn(x) for x in x]
dft_signal1 = dft_transform(fx)
k1 = range(N)
fft_signal1 = np.fft.fft(fx)
# dft_plotting(k1,dft_signal1,x,fx,"Manual")
# dft_plotting(k1,fft_signal1,x,fx,"inbuilt")


#g = e^(-t^2)
gn= lambda t: exp(-t**2)
a,b = 0,1
N = 200 
t = np.linspace(a,b,N)
gt = [ gn(t) for t in t ]
dft_signal2 = dft_transform(gt)
k2 = range(N)
fft_signal2 = np.fft.fft(gt)
# dft_plotting(k2,dft_signal2,t,gt,"Manual")
# dft_plotting(k2,fft_signal2,t,gt,"inbuilt")

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
# error1 = compare_fft(k1,fft_signal1,dft_signal1)
# error2 = compare_fft(k2,fft_signal2,dft_signal2)
# print(f"error between dft and fft is {error1} of function f=3sin(2pit) + sin(8pit) + 0.5sin(14pit) \nerror between dft and fft is {error2} of function f = e^(-t^2)")
    
    
    
##########################################################################################################################



#reflection rotation translation scaling 
# original complex numbers
z = [2+1j, -3+2j, -1-3j]

# transformations
def y_reflection(z): return [-zi.real + zi.imag*1j for zi in z]
def x_reflection(z): return [zi.real - zi.imag*1j for zi in z]
def reflection_origin(z): return [-zi.real - zi.imag*1j for zi in z]
def translation(z, a, b): return [(zi.real+a) + (zi.imag+b)*1j for zi in z]
def scaling(z, a, b): return [(zi.real*a) + (zi.imag*b)*1j for zi in z]

#rotation
def rotation(z, theta):
    angle = M.radians(theta)
    ans = []
    for zi in z:
        x = zi.real*M.cos(angle) - zi.imag*M.sin(angle)
        y = zi.real*M.sin(angle) + zi.imag*M.cos(angle)
        ans.append(x + y*1j)
    return ans

#rotation and translation
def rotation_and_translation(z, theta, a, b):
    return translation(rotation(z, theta), a, b)

# plotting function (triangle to triangle)
def plot_transform(z_before_list, z_after_list, title="Transformation"):
    # close the triangles by repeating first point
    zb_real = [p.real for p in z_before_list] + [z_before_list[0].real]
    zb_imag = [p.imag for p in z_before_list] + [z_before_list[0].imag]

    za_real = [p.real for p in z_after_list] + [z_after_list[0].real]
    za_imag = [p.imag for p in z_after_list] + [z_after_list[0].imag]

    # plot original triangle
    plt.plot(zb_real, zb_imag, "o-b", label="Original Triangle")

    # plot transformed triangle
    plt.plot(za_real, za_imag, "o-r", label="Transformed Triangle")

    # axis setup
    plt.axhline(0, color="black", linewidth=1.2)
    plt.axvline(0, color="black", linewidth=1.2)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


# Example runs
# plot_transform(z, translation(z, 2, 3), "Translation (a=2, b=3)")
# plot_transform(z, rotation(z, 30), "Rotation (30°)")
# plot_transform(z, scaling(z, 2, 3), "Scaling (a=2, b=3)")
# plot_transform(z, y_reflection(z), "Reflection about Y-axis")
# plot_transform(z, x_reflection(z), "Reflection about X-axis")
# plot_transform(z, reflection_origin(z), "Reflection about Origin")
# plot_transform(z, rotation_and_translation(z, 30, 2, 3), 
#                "Rotation + Translation (θ=30°, a=2, b=3)")

def z_rotate(z,theta):
    z_rot =[]
    for zi in z :
        r = abs(zi)
        angle = (phase(zi) + M.radians(theta))
        z_rotated = r*exp(1j*angle)
        z_rot.append(z_rotated)
    return z_rot
# plot_transform(z, z_rotate(z, 30), "Rotation (30°) re^(t1 + t2)i")




#cirlce of radius 1 |z| = (x^2 + y ^2)^(1/2) = r = 1 so z = (1)e^(i*theta)
r = 1
theta = np.linspace(0,2*M.pi,100)
z = r*np.exp(1j*theta)
z_real = z.real
z_imag = z.imag
plt.plot(z_real ,z_imag,label="circle")
plt.plot([0,z_real[10]],[0,z_imag[10]],color="red",label="radius")
plt.plot([0,z_real[0]],[0,z_imag[0]],color="red")
plt.title("circle of radius 1 |z| = 1")
plt.legend()
plt.grid(True)
plt.xlabel("real axis")
plt.ylabel("imaginary axis")
plt.axis("equal")
plt.show()


#######################################################################################################################



#integral 
# ------------------------
# 2-point Gauss Quadrature using for i in range
# ------------------------
def gauss2(fn, a, b):
    nodes = [-1/M.sqrt(3), 1/M.sqrt(3)]  # 2-point nodes
    weights = [1, 1]                  # 2-point weights

    m = (b - a) / 2
    n = (a + b) / 2

    result = 0
    for i in range(len(nodes)):
        xi = nodes[i]
        wi = weights[i]
        x = m*xi + n       # map node to [a,b]
        result += wi * fn(x)
    return m * result      # multiply by (b-a)/2

# ------------------------
# 3-point Gauss Quadrature using for i in range
# ------------------------
def gauss3(fn, a, b):
    nodes = [-M.sqrt(3/5), 0, M.sqrt(3/5)]  # 3-point nodes
    weights = [5/9, 8/9, 5/9]           # 3-point weights

    m = (b - a) / 2
    n = (a + b) / 2

    result = 0
    for i in range(len(nodes)):
        xi = nodes[i]
        wi = weights[i]
        x = m*xi + n
        result += wi * fn(x)
    return m * result


fn = lambda x: 5*x**3 - 4*x**2 + 7*x + 10
gn = lambda x: x*M.exp(M.cos(x) + M.sin(x))
# 2-point Gauss
# print("2-point Gauss:", gauss2(fn, 2, 5))

# 3-point Gauss
# print("3-point Gauss:", gauss3(gn, 0, 1))



#simpson integration
import math as M
def simpson_integration(f, a, b, n):
    if n % 2 == 1:
       print("Number of subintervals must be even.")
       return 

    h = (b - a) / n
    x = a
    integral = f(x) + f(b)

    for i in range(1, n):
        x += h
        if i % 2 == 0:
            integral += 2 * f(x)
        else:
            integral += 4 * f(x)

    integral *= h / 3
    return integral
def example_function(x):  
    return M.log(M.sin(x))
a =float(input("enter the lower limit :"))
b =float(input("enter the upper limit :"))
n = int(input("enter the number of iterations :"))
result = simpson_integration(example_function, a, b, n)
print("The integral is:", result)

######################################################################################################################


# Coefficients
pn = lambda x: 0
qn = lambda x: 0
rn = lambda x: -9.8

# Domain and boundary conditions
a, b = 0, 5
y0, y5 = 50, M.sqrt(10)

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
plt.plot(x_, y_exact, label="Analytic Solution", linestyle='--')
plt.plot(x, y_fdm, label="Finite Difference Solution")
plt.xlabel("time(in seconds)")
plt.ylabel("(displacement(in m))")
plt.title("FDM vs Analytic Solution")
plt.legend()
plt.grid()
plt.show()



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
plt.plot(t, y(t))
plt.scatter([0, 5], [y(0), y(5)], color="red")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid(True)
plt.show()