import cmath as M
import matplotlib.pyplot as plt
import numpy as np

# exp 1
z1 = 2+3j
z2 = 2-5j
print(f"addition is {z1+z2}")
print(f"subtraction is {z1-z2}")
print(f"multiplication is {z1*z2}")
print(f"division is {z1/z2}")
z3 = -3 -1j 

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
print(M.phase(z3))
        
 # exp 2
z1 = 2 + 3j
z2 = 2-5j
def nth_root(n,z):
    roots =[]
    r = abs(z)
    theta = M.phase(z)
    for i in range(n):
        angle = (theta + 2*M.pi*i)/n
        root= pow(r,1/n)*M.exp(1j*angle)
        roots.append(root)
    return roots
root1=nth_root(3,z1)
root2=nth_root(5,z2)

# --------- Plot for first number ----------
plt.figure(figsize=(6,6))
for idx, num in enumerate(root1, 1):
    plt.plot([0, num.real], [0, num.imag], label=f"Root {idx}")
plt.title("Cube roots of z1 = 2+3j")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.legend()
plt.grid(True)
plt.axis("equal")

# --------- Plot for second number ----------
plt.figure(figsize=(6,6))
for idx, num in enumerate(root2, 1):
    plt.plot([0, num.real], [0, num.imag], label=f"Root {idx}")
plt.title("Fifth roots of z2 = 2-5j")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.legend()
plt.grid(True) 
plt.axis("equal")
plt.show()


# exp 3
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
    