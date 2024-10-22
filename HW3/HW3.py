import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def partB():
    def q(x):
        return x**2

    L = 4
    dx = .1
    xspan = np.arange(-1*L, L+dx, dx)
    epsilons = []
    eigenfunctions = []
    N = len(xspan)

    A = np.matrix(np.zeros((N, N)))

    for i in range(1,N - 1):
        A[i,i] = 2/dx**2 + q(xspan[i])
        A[i,i + 1] = -1/(dx**2)
        A[i,i - 1] = -1/(dx**2)

    A[0,0] = 3
    A[0,1] = -4
    A[0,2] = 1

    A[-1,-1] = -3
    A[-1, -2] = 4
    A[-1, -3] = -1

    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    for i in range(len(eigenvalues)): #find first positive eigenvalue
        if eigenvalues[i] > 0:
            break

    epsilons = eigenvalues[i:i+5] #get first 5 positive eigenvalues
    for j in range(i, i+6):
        arrEigenvector = np.squeeze(np.asarray(eigenvectors[:,j]))
        normedEigenfunction = eigenvectors[:,j]/np.sqrt(np.trapezoid(arrEigenvector**2))
        plt.plot(xspan, normedEigenfunction)
        eigenfunctions.append(abs(normedEigenfunction))
    
    plt.legend(epsilons)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solution Plot')
    plt.grid(True)
    plt.show()


def partC():
    def shoot(y, x, epsilon, gamma):
        return [y[1], y[0]*(gamma*abs(y[0]) + x**2 - epsilon)]

    xspan = [-4, 4]  # x range
    # initial derivative value
    # step size for derivative adjustment
    epsilon = 0
    epsilons = []
    eigenfunctions = []
    gamma = 0.05

    for n in range(2):
        
        dEpsilon = 0.1
        for j in range(1000):
            y0 = [1, np.sqrt(gamma + 16 - epsilon)]  # initial condition
            x = np.arange(xspan[0], xspan[1] + .1, .1)  # grid for odeint
            ysol = odeint(shoot, y0, x, (epsilon, gamma))  # solve ODE
            
            if abs(ysol[-1, 1] + np.sqrt(gamma*abs(ysol[-1,0]) + 16 - epsilon) * ysol[-1, 0]) < 1e-6:  # check convergence
                break
            
            if (-1) ** (n+1) * (ysol[-1, 1] + (np.sqrt(gamma*abs(ysol[-1,0]) + 16 - epsilon) * ysol[-1,0])) < 0:
                epsilon += dEpsilon
            else:
                epsilon -= dEpsilon
                dEpsilon /= 2
        normedEigenfunction = ysol[:, 0]/np.sqrt(np.trapezoid(ysol[:,0]**2, x))
        eigenfunctions.append(abs(normedEigenfunction))
        epsilons.append(epsilon)
        plt.plot(x, normedEigenfunction)
        epsilon += 1

    A1 = np.matrix(np.transpose(eigenfunctions))
    A2 = np.array(epsilons)

    plt.legend(epsilons)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solution Plot')
    plt.grid(True)
    plt.show()


partB()