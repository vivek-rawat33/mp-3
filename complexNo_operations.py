# import cmath 
# import math as M
# import matplotlib.pyplot as plt
# import numpy as np
# #reflection about y axis
# # z1 = 2+1j
# # z2 = -3 + 2j
# # z3 = -1 - 3j
# z = [2+1j, -3+2j, -1-3j]

# # reflection about y axis x+iy -> -x + iy
# def y_reflection(z):
#     ans = []
#     for z in z:
#          ans.append(-z.real + z.imag*1j)
#     return ans
# print(y_reflection(z))


# # about x axis
# def x_reflection(z):
#     ans = []
#     for z in z:
#         ans.append(z.real - z.imag*1j)
#     return ans
# print(x_reflection(z))

# # about origin
# def reflection_origin(z):
#     ans =[]
#     for z in z:
#         ans.append(-z.real - z.imag*1j)
#     return ans
# print(reflection_origin(z))


# # translation of complex number 
# def translation(z,a,b):
#     ans =[]
#     for z in z:
#         real = z.real + a
#         imag = z.imag+ b
#         ans.append(real + imag*1j)
#     return ans
# print(translation(z,2,3))

# #scaling
# def scaling(z,a,b):
#     ans =[]
#     for z in z:
#         real = z.real*a
#         imag = z.imag*b
#         ans.append(real + imag*1j)
#     return ans 
# print(scaling(z,3,4))

# #rotation (theta angle)
# def rotation(z,theta):
#     ans = []
#     for z in z:
#         angle = M.radians(theta)
#         x = z.real*M.cos(angle) - z.imag*M.sin(angle)
#         y = z.real*M.cos(angle) + z.imag*M.sin(angle)
#         ans.append( x + y*1j)
#     return ans
# print("rotation")
# print(rotation(z, 45))

# # rotation + translation
# def rotation_and_translation(z_list, theta, a, b):
#     rotated = rotation(z_list, theta)           
#     translated = translation(rotated, a, b)     
#     return translated



# def plot_transform(z_before_list, z_after_list, title="Transformation"):
#     for i in range(len(z_before_list)):
#         zb = z_before_list[i]
#         za = z_after_list[i]

#         # Original point
#         plt.scatter(zb.real, zb.imag, color="blue", label="Original" if i==0 else "")
#         plt.plot([0, zb.real], [0, zb.imag], "--", color="blue")
#         plt.text(zb.real+0.1, zb.imag+0.1,
#                  f"({zb.real},{zb.imag})", color="black")

#         # Transformed point
#         plt.scatter(za.real, za.imag, color="red", label="Reflected" if i==0 else "")
#         plt.plot([0, za.real], [0, za.imag], "--", color="red")
#         plt.text(za.real+0.1, za.imag+0.1,
#                  f"({za.real},{za.imag})", color="black")
#         plt.plot([zb.real, za.real], [zb.imag, za.imag], "--", color='green')

    
#     plt.axhline(0, color="black", linewidth=1.5)  
#     plt.axvline(0, color="black", linewidth=1.5) 
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# plot_transform(z, y_reflection(z), "Reflection about Y-axis")
# plot_transform(z,x_reflection(z),"Reflection about X-axis" )
# plot_transform(z,reflection_origin(z),"Reflection about Origin")
# plot_transform(z,translation(z,2,3),"Translation a=2 , b=3")
# plot_transform(z,rotation(z,30),"Rotation by 30 degree")
# plot_transform(z,scaling(z,2,3),"scaling a=2 , b=3")
# plot_transform(z,rotation_and_translation(z,30,2,3),"rotation translation by theta =30 degree and a = 2 , b = 3")

# # circel |z| = 1 , z = x + iy, |z| = (x^2 + y^2) ^ (1/2)

# # Parameters
# a, b = 2, 3

# # Generate circle points (|z| = 1)
# theta = np.linspace(0, 2*np.pi, 400)
# x = np.cos(theta)   # x = cosθ
# y = np.sin(theta)   # y = sinθ

# # Scaled points (ellipse)
# x_scaled = a * x
# y_scaled = b * y

# # Plotting
# plt.plot(x, y, label="|z| = 1 (Unit Circle)", color="blue")
# plt.plot(x_scaled, y_scaled, label=f"Scaled (a={a}, b={b})", color="red")

# # Axes
# plt.axhline(0, color="black", linewidth=1.2)
# plt.axvline(0, color="black", linewidth=1.2)

# plt.gca().set_aspect("equal", adjustable="box")  # Equal aspect ratio
# plt.legend()
# plt.grid(True)
# plt.title("Scaling of Unit Circle in Complex Plane")
# plt.show()



import math as M
import matplotlib.pyplot as plt
import numpy as np

# original complex numbers
z = [2+1j, -3+2j, -1-3j]

# transformations
def y_reflection(z): return [-zi.real + zi.imag*1j for zi in z]
def x_reflection(z): return [zi.real - zi.imag*1j for zi in z]
def reflection_origin(z): return [-zi.real - zi.imag*1j for zi in z]
def translation(z, a, b): return [(zi.real+a) + (zi.imag+b)*1j for zi in z]
def scaling(z, a, b): return [(zi.real*a) + (zi.imag*b)*1j for zi in z]

def rotation(z, theta):
    angle = M.radians(theta)
    ans = []
    for zi in z:
        x = zi.real*M.cos(angle) - zi.imag*M.sin(angle)
        y = zi.real*M.sin(angle) + zi.imag*M.cos(angle)
        ans.append(x + y*1j)
    return ans

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
plot_transform(z, translation(z, 2, 3), "Translation (a=2, b=3)")
plot_transform(z, rotation(z, 30), "Rotation (30°)")
plot_transform(z, scaling(z, 2, 3), "Scaling (a=2, b=3)")
plot_transform(z, y_reflection(z), "Reflection about Y-axis")
plot_transform(z, x_reflection(z), "Reflection about X-axis")
plot_transform(z, reflection_origin(z), "Reflection about Origin")
plot_transform(z, rotation_and_translation(z, 30, 2, 3), 
               "Rotation + Translation (θ=30°, a=2, b=3)")
