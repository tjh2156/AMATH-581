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
        print()

        if abs(fc) < 1e-6:
            break
    A1 = x
    np.save("A1.npy", A1)

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
    np.save("A2.npy", A2)

    #A3 creation
    A3 = np.array([len(A1), len(A2)])
    np.save("A3.npy", A3)


def Q2():
    A = np.matrix([[1,2], [-1, 1]])
    B = np.matrix([[2,0],[0,2]])
    C = np.matrix([[2,0,-3],[0,0,-1]])
    D = np.matrix([[1,2], [2,3], [-1,0]])
    x = np.matrix([[1], [0]])
    y = np.matrix([[0], [1]])
    z = np.matrix([[1], [2], [-1]])

    A4 = A + B
    np.save("A4.npy", A4)

    A5 = 3*x - 4*y
    np.save("A5.npy", A5)

    A6 = A * x
    np.save("A6.npy", A6)

    A7 = B*(x - y)
    np.save("A7.npy", A7)

    A8 = D*x
    np.save("A8.npy", A8)

    A9 = D*y + z
    np.save("A9.npy", A9)

    A10 = A*B
    np.save("A10.npy", A10)

    A11 = B*C
    np.save("A11.npy", A11)

    A12 = C*D
    np.save("A12.npy", A12)

Q1()
Q2()