# # Gauss quadrature method for integration
import math as M
# def func(x):
#     return 5*x**3 - 4*x**2+ 7*x +10

# def gauss_quadrature(a, b):
#     points = [-0.577, 0.577]
#     weights = [1, 1]

#     def to_interval(x):
#         return ((b - a) / 2) * x + (a + b) / 2

#     total = 0
#     for i in range(2):
#         x_val = to_interval(points[i])
#         fx = func(x_val)  
#         total += weights[i] * fx
#     return ((b - a) / 2) * total

# result = gauss_quadrature(2, 5)
# print("Result:", result)

# a = int(input("Enter the lower limit :"))
# b = int(input("Enter the upper limit :"))

# def func(x):
#     return 5*x**3 - 4*x**2 + 7*x + 10

# def integral():
#     x1 = 0.577
#     x2 = -0.577
#     w1 = w2 = 1

#     p1 = ((b - a) / 2) * x1 + (a + b) / 2
#     p2 = ((b - a) / 2) * x2 + (a + b) / 2

#     ans = ((b - a) / 2) * (w1 * func(p1) + w2 * func(p2))
#     return ans

# result =integral()
# print(result)


a = int(input("Enter the lower limit :"))
b = int(input("Enter the upper limit :"))

def func(x):
    return x**5 - 10*x**4 + 4*x**3 +  3*x**2+ 10

def integral():
    x1 = -M.sqrt(3/5)
    x2 = 0
    x3 = M.sqrt(3/5)
    w1 = 5/9
    w2 = 8/9
    w3 = 5/9

    p1 = ((b - a) / 2) * x1 + (a + b) / 2
    p2 = ((b - a) / 2) * x2 + (a + b) / 2
    p3 = ((b - a) / 2) * x3 + (a + b) / 2

    ans = ((b - a) / 2) * (w1 * func(p1) + w2 * func(p2) + w3 * func(p3))
    return ans

result =integral()
print(result)