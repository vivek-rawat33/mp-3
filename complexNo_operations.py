
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
