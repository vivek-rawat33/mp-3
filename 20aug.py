# #To plot the complex function
from cmath import *
from matplotlib.pyplot import *
from numpy import *

x2= linspace(-1,2,100)
y2= linspace(-2,3,100)

x1= linspace(-10,10,100)
y1= linspace(-10,10,100)


z1= [complex(x1[i],y1[i]) for i in range(len(x1))]
z2= [complex(x2[i],y2[i]) for i in range(len(x2))]

def fplot(x,y):
    n= len(x)
    l= []
    for i in range(n):
        u= x[i]**2 - y[i]**2
        v= 2*x[i]*y[i]
        l.append(complex(u,v))
    return l

pts1= fplot(x1,y1)
pts2= fplot(x2,y2)

def cplot(r,z):
    x1 = [i.real for i in r]
    y1 = [i.imag for i in r]
    x2 = [i.real for i in z]
    y2 = [i.imag for i in z]
    
    plot(x1, y1, label='f(z)',color='blue')
    plot(x2, y2,color='black')
    scatter([x2[0], x2[-1]], [y2[0], y2[-1]],color="red",label=f'z\nz1:{z[0]}\nz2:{z[-1]}')
    for i in range(len(r)):
        plot([x2[i],x1[i]],[y2[i],y1[i]],linestyle='--',color='grey')
    xlabel("real axis")
    ylabel("imaginary axis")
    legend()
    grid()
    show()

cplot(pts1,z1)
cplot(pts2,z2)

# #To transform complex numbers as 2-D vectors e.g. translation
# z1= [(2+1j),(-3+2j),(-1-3j)]
# a,b= 2,3

# def trans(z,a,b):
#     ztrans= []
#     for i in z:
#         tz= complex((i.real+a),(i.imag + b))
#         ztrans.append(tz)
#     x=[i.real for i in z]
#     y= [i.imag for i in z]
#     xt= [i.real for i in ztrans]
#     yt= [i.imag for i in ztrans]

#     scatter(x, y, color='brown',label='z')
#     scatter(xt, yt, color='green',label=f'z translated by ({a},{b})')
 
#     for i in range(len(x)):
#         plot([x[i],x[i-1]],[y[i],y[i-1]],color='orange')

#     for i in range(len(xt)):
#         plot([xt[i],xt[i-1]],[yt[i],yt[i-1]],color='red')

#     for i in range(len(x)):
#             plot([x[i],xt[i]],[y[i],yt[i]],linestyle='--',color='grey')

    # xlabel("real axis")
    # ylabel("imaginary axis")
    # legend()
    # grid()
    # show()

# trans(z1,a,b) 


#To transform complex numbers as 2-D vectors e.g. scaling

# z=[]
# t= np.linspace(0,2*pi,200)
# for i in t:
#     z.append(complex(cos(i),sin(i)))
# def cscale(z,a,b,lbl=""):
#     x1= [i.real*a for i in z]
#     y1= [i.imag*b for i in z]
#     plot(x1,y1,label=f"{lbl}")

# cscale(z,1,1,"circle: |x+jy|=1")
# cscale(z,2,3,"scaled by (2,3)")
# legend()
# xlabel(f"real axis")
# ylabel(f"imaginary axis")
# axis("equal")   
# grid()
# show()

#To transform complex numbers as 2-D vectors e.g. rotation

# z= [(2+1j),(-3+2j),(-1-3j)]

# def crotate(z,theta):
#     r= abs(z)
#     theta0= phase(z)

#     c= r*complex(cos(theta0),sin(theta0))
#     scatter(c.real,c.imag,label=f'original:{z}')
#     c1= r*complex(cos(theta0+theta),sin(theta0+theta))
#     scatter(c1.real,c1.imag,label=f'angle shift by {theta}')
#     plot([0,c1.real],[0,c1.imag],color='grey')
#     plot([0,c.real],[0,c.imag],color='grey')
# for i in z:
#     crotate(i,pi/6)
# xlabel(f"real axis")
# ylabel(f"imaginary axis")
# axhline(0)
# axvline(0)
# legend()
# grid()
# show()

