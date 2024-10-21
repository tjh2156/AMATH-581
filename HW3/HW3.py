import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def partB():
    def q(x, epsilon):
        return x**2 - epsilon

    def leftY(L, epsilon):
        return np.exp(-1*L*np.sqrt(L**2 - epsilon))

    def rightY(L, epsilon):
        return np.exp(-1*L*np.sqrt(L**2 - epsilon))

    L = 4
    dx = .1
    xspan = np.arange(-1*L, L+dx, dx)
    epsilon = 0
    epsilons = []
    eigenfunctions = []
    N = len(xspan)

    A = np.matrix(np.zeros((N-2, N-2)))
    b = np.matrix((np.zeros((N-2, 1))))

    for n in range(5):
        dEpsilon = .1
        for j in range(1000):
            #set up b
            b[0] = leftY(L, epsilon)
            b[-1] = rightY(L, epsilon)

            #set up A
            for i in range(len(A)):
                A[i,i] = 2 + dx**2 * q(xspan[i+1], epsilon)
                if i >= 1:
                    A[i, i-1] = -1
                if i < N-3:
                    A[i, i+1] = -1

            #Solve Ax=b
            ysol = np.linalg.solve(A, b)

            #BCs
            derivativeLeft = (-3*leftY(L, epsilon) + 4*ysol[0] - ysol[1])/(2*dx)
            derivativeRight = (3*rightY(L, epsilon) - 4*ysol[-1] + ysol[-2])/(2*dx)
            if abs(derivativeLeft) < 1e-6:
                print("found it ", j)
                break
            if (-1) ** (n+1) * (derivativeLeft) < 0:
                epsilon += dEpsilon
            else:
                epsilon -= dEpsilon
                dEpsilon /= 2
        print(f"final j = {j}   epsilon num = {n}     dEpsilon = {dEpsilon}     epsilon = {epsilon}")
        ysol = np.squeeze(np.asarray(ysol))
        ysol = np.array([leftY(L, epsilon), *ysol, rightY(L, epsilon)])
        normalizedY = ysol/np.sqrt(np.trapezoid(ysol**2, dx=dx))
        plt.plot(xspan, normalizedY)
        epsilons.append(epsilon)
        eigenfunctions.append(normalizedY)
        epsilon += 1

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


partC()