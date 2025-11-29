
import numpy as np
import matplotlib.pyplot as plt

z = [2+1j, -3+2j, -1-3j]

def cref(z, m=1, c=0):
    x, y = z.real, z.imag
    a, b = -m, 1
    ref = 2*(a*x + b*y + c)/(a**2 + b**2)

    # Reflections
    x_xaxis, y_xaxis = x, -y
    x_yaxis, y_yaxis = -x, y
    x_line, y_line = x - a*ref, y - b*ref

    # Original
    plt.scatter(x, y,label=f"original point: z={z}")
    plt.text(x+0.1, y+0.1, f"{z}", fontsize=9)

    # Reflections
    plt.scatter(x_xaxis, y_xaxis)
    plt.text(x_xaxis+0.1, y_xaxis+0.1, f"{complex(x_xaxis,y_xaxis)}", fontsize=9)

    plt.scatter(x_yaxis, y_yaxis)
    plt.text(x_yaxis+0.1, y_yaxis+0.1, f"{complex(x_yaxis,y_yaxis)}", fontsize=9)
    plt.scatter(x_line, y_line)
    plt.text(x_line+0.1, y_line+0.1, f"{complex(x_line,y_line)}", fontsize=9)
    # Connecting dashed lines
    plt.plot([x, x_xaxis], [y, y_xaxis], "--")
    plt.plot([x, x_yaxis], [y, y_yaxis], "--")
    plt.plot([x, x_line], [y, y_line], "--")
m, c = 1, 0  # slope & intercept of reflection line

# Line of reflection
t = np.linspace(-2.5, 3, 200)
plt.plot(t, m*t + c, "k--", label=f"y={m}x+{c}")

# Apply to each point
for i in z:
    cref(i, m, c)
plt.axhline(0, color="blue")
plt.axvline(0, color="blue")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()


#To integrate using gauss.....

from math import *

fn= lambda x: 5*x**3 - 4*x**2 +7*x +10
f1= lambda x: 5*x**3
f2= lambda x: 4*x**2
f3= lambda x: 7*x
f4= lambda x:10

def integ(fn,a,b):
    m= (b-a)/2
    n= (b+a)/2
    t=1/sqrt(3)

    return (m*(fn(m*t+n)+fn(-m*t +n)))

ans= integ(fn,-1,1)
print(ans)
ans_= integ(fn,2,5)
print(ans_)
print("cross-checking with separate formulas: ")
print(integ(f1,-1,1)-integ(f2,-1,1)+integ(f3,-1,1)+integ(f4,-1,1))

print(integ(f1,2,5)-integ(f2,2,5)+integ(f3,2,5)+integ(f4,2,5))

f5= lambda x: x**5 -10*x**4 +4*x**3 +3*x**2 +10
f6= lambda x: x**5 
f7= lambda x: -10*x**4
f8= lambda x: 4*x**3 
f9= lambda x: 3*x**2 
f10= lambda x: 10
def integ2(fn,a,b):
    m= (b-a)/2
    n= (b+a)/2
    t= sqrt(3/5)
    return (m*((5/9)*fn(m*t+n)+(5/9)*fn(-m*t +n)+ (8/9)*fn(n)))

print(integ2(f5,2,5))
print("verification: ")
print(integ2(f6,2,5)+integ2(f7,2,5)+integ2(f8,2,5)+integ2(f9,2,5)+integ2(f10,2,5))