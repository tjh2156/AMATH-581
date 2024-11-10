import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

def partA(L, n):
    N = n * n
    dx = L*2/n

    e0 = np.zeros((N, 1))  # vector of zeros
    e1 = np.ones((N, 1))   # vector of ones
    e2 = np.copy(e1)    # copy the one vector
    e4 = np.copy(e0)    # copy the zero vector

    for j in range(1, n+1):
        e2[n*j-1] = 0  # overwrite every m^th value with zero
        e4[n*j-1] = 1  # overwirte every m^th value with one

    # Shift to correct positions
    e3 = np.zeros_like(e2)
    e3[1:N] = e2[0:N-1]
    e3[0] = e2[N-1]

    e5 = np.zeros_like(e4)
    e5[1:N] = e4[0:N-1]
    e5[0] = e4[N-1]

    # Place diagonal elements
    diagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
                e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
                e4.flatten(), e1.flatten(), e1.flatten()]
    offsets = [-(N-n), -n, -n+1, -1, 0, 1, n-1, n, (N-n)]

    secondDegreeXY = spdiags(diagonals, offsets, N, N).toarray()/dx**2

    plt.figure(1)
    plt.spy(secondDegreeXY)
    plt.title('Second Degree in X and Y Matrix Structure')

    firstDegreeX = spdiags([e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()], [-(N-n),-n,n, (N-n)], N, N).toarray()/(2*dx)
    plt.figure(2)
    plt.spy(firstDegreeX)
    plt.title('First Degree in X Matrix Structure')

    e6 = np.copy(e1)
    e7 = np.copy(e1)
    for j in range(n):
        e6[j*n] = 0
        e7[j*n - 1] = 0

    firstDegreeY = spdiags([-e7.flatten(), e6.flatten()], [-1,1], N, N).toarray()

    for j in range(n):
        firstDegreeY[j*n, (j+1)*n - 1] = -1
        firstDegreeY[(j+1)*n - 1, j*n] = 1

    firstDegreeY /= 2*dx

    plt.figure(3)
    plt.spy(firstDegreeY)
    plt.title('First Degree in Y Matrix Structure')
    plt.show()

    return secondDegreeXY, firstDegreeX, firstDegreeY

A1, A2, A3 = partA(10, 8)