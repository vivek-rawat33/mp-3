import cmath as M
import matplotlib.pyplot as plt
import numpy as np
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


# def z_func():
#     x= np.linspace(2,-7,15)
#     y= np.linspace(3,-9,15)
#     z=[]
#     for i in range(len(x)):
#         ope= x[i]**2 -y[i]**2 + 2*x[i]*y[i]*1j
#         z.append(ope)
#     return z
# z_result = z_func()
# for num in z_result:
#     plt.plot([0,num.real],[0,num.imag],color="blue")
    
# plt.grid(True)
# plt.show()

# def fz(x, y):
#     return x**2 - y**2 + 2*x*y*1j

# x_vals = np.linspace(-1, 2, 100)
# y_vals = np.linspace(-2, 3, 100)

# x1 = np.linspace(-10,10,100)
# y1 = np.linspace(-10,10,100)
# Z = fz(x1, y1)

# # for z in Z:
#     # plt.plot([0, z.real], [0, z.imag], color='gray')
# plt.plot(Z.real, Z.imag, color='blue', label='f(z)')
# plt.plot([-10,10], [-10,10], 'r', label='z1:(-10+10j)')


# plt.grid(True, linestyle='--')

# plt.xlabel('real axis')
# plt.ylabel('imaginary axis')
# plt.legend()
# plt.show()
