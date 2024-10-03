import numpy as np

def Q1():
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
    A1 = x[1:]

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
    A3 = np.array([len(A1), len(A2)])


def Q2():
    A = np.matrix([[1,2], [-1, 1]])
    B = np.matrix([[2,0],[0,2]])
    C = np.matrix([[2,0,-3],[0,0,-1]])
    D = np.matrix([[1,2], [2,3], [-1,0]])
    x = np.matrix([[1], [0]])
    y = np.matrix([[0], [1]])
    z = np.matrix([[1], [2], [-1]])

    A4 = A + B

    A5 = 3*x - 4*y

    A6 = A * x

    A7 = B*(x - y)

    A8 = D*x

    A9 = D*y + z

    A10 = A*B

    A11 = B*C

    A12 = C*D

Q1()
Q2()