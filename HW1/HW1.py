import numpy as np

A1 = None
A2 = None
A3 = None

#Newton-Raphson
x = np.array([-1.6])
for i in range(1000) :
    fc = x[i] * np.sin(3*x[i]) - np.exp(x[i])
    fcPrime = np.sin(3*x[i]) + 3*x[i]*np.cos(3*x[i]) - np.exp(x[i])
    xNext = x[i] - fc/fcPrime
    x = np.append(x, xNext)

    if abs(fc) < 1e-6:
        break
A1 = x

#Bisection
left = -.7
right = -.4
midpoints = np.array([])
for i in range(100):
    midpoints = np.append(midpoints, (left + right)/2)
    fc = midpoints[i] * np.sin(3*midpoints[i]) - np.exp(midpoints[i])
    if fc > 0:
        left = midpoints[i]
    else:
        right = midpoints[i]
    
    if abs(fc) < 1e-6:
        break
A2 = midpoints

#A3 creation
A3 = np.array([len(A1)-1, len(A2)])


A = np.matrix([[1,2], [-1, 1]])
B = np.matrix([[2,0],[0,2]])
C = np.matrix([[2,0,-3],[0,0,-1]])
D = np.matrix([[1,2], [2,3], [-1,0]])
x = np.matrix([[1], [0]])
y = np.matrix([[0], [1]])
z = np.matrix([[1], [2], [-1]])

A4 = A + B

A5 = np.array(3*x - 4*y).flatten()

A6 = np.array(A * x).flatten()

A7 = np.array(B*(x - y)).flatten()

A8 = np.array(D*x).flatten()

A9 = np.array(D*y + z).flatten()

A10 = A*B

A11 = B*C

A12 = C*D

print(A1.shape)
print(A5.shape)
print(A6.shape)
print(A7.shape)
print(A8.shape)
print(A9.shape)