import numpy as np
from scipy.integrate import solve_ivp
from scipy import integrate
import matplotlib.pyplot as plt

def setupMatrices(dx, N):
    #AssumeN anddxgiven
        A= np.zeros((N,N))
        for j in range(N-1):
            A[j,j +1]= 1
            A[j+ 1,j]=-1
        A1=A /(2* dx)#Dirichletmatrices

        A2=np.copy(A)
        A2[0,0]=-4/3
        A2[0,1]= 4/ 3
        A2[N- 1,N- 1]=4 /3
        A2[N- 1,N- 2]=-4 /3
        A2=A2/ (2* dx)# Neumannmatrices

        A3=np.copy(A)
        A3[N- 1,0]= 1
        A3[0,N- 1]=-1
        A3=A3/ (2* dx)# PeriodicBC matrices

        B = np.zeros((N, N))
        for j in range(N):
            B[j, j] =-2
        for j in range(N- 1):
            B[j, j + 1] = 1
            B[j + 1, j] = 1
        B1 = B / (dx**2) # Dirichlet matrices for B

        B2 = np.copy(B)
        B2[0, 0] =-2 / 3
        B2[0, 1] = 2 / 3
        B2[N- 1, N- 1] =-2 / 3
        B2[N- 1, N- 2] = 2 / 3
        B2 = B2 / (dx**2) # Neumann matrices for B

        B3 = np.copy(B)
        B3[N- 1, 0] = 1
        B3[0, N- 1] =-1
        B3 = B3 / (dx**2) # Periodic BC matrices for B

        return A1, A2, A3, B1, B2, B3

def partB():

    L = 4 # domain size
    dx = .1
    x = np.arange(-L, L+dx, dx) # add boundary points
    N = 79

    A = np.zeros((N, N)) # Compute P matrix
    for j in range(N):
        A[j,j] = (-2- x[j+1]**2 * dx**2)/(-dx**2)
        if j > 0:
            A[j,j-1] = 1/(-dx**2)
        if j < N - 1:
            A[j,j+1] = 1/(-dx**2)
    
    A[0,0] = 4/3/(-dx**2)
    A[0,1] = -1/3/(-dx**2)
    A[-1,-1] = 4/3/(-dx**2)
    A[-1,-2] = -1/3/(-dx**2)
    
    D, V = np.linalg.eig(A)

    sorted_indices=np.argsort(np.abs(D))[::-1]
    Dsort=D[sorted_indices]
    Vsort=V[:,sorted_indices]
    
    D5 = np.abs(Dsort[N-5:N])
    V5 = Vsort[:,N-5:N]

    A3 = [] #eigenfunctions
    A4 = D5 #eigenvalues

    for k in range(5):
        arrEigenvector = np.squeeze(np.asarray(V5[:,k]))

        leftBoundary = 4/3 * arrEigenvector[0] - 1/3 * arrEigenvector[1]
        rightBoundary = 4/3 * arrEigenvector[-1] - 1/3 * arrEigenvector[-2]
        arrEigenvector = np.array([leftBoundary, *arrEigenvector, rightBoundary])

        normedEigenfunction = arrEigenvector/np.sqrt(np.trapezoid(arrEigenvector**2, dx=dx))
        plt.plot(x, normedEigenfunction)
        A3.append(abs(normedEigenfunction))

    A3 = np.array(A3).T
    print(A3.shape)

    plt.legend(np.asarray(D5, float))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solution Plot')
    plt.grid(True)
    plt.show()

    return A3, A4


def partC(L, dx, gammas):
    def shoot(x, y, epsilon, gamma):
        return [y[1], y[0]*(gamma*abs(y[0]) + x**2 - epsilon)]

    xspan = [-L, L+dx]  # x range
    # initial derivative value
    # step size for derivative adjustment
    positiveGammaEpsilons = []
    negativeGammaEpsilons = []
    positiveGammaEigenfunctions = []
    negativeGammaEigenfunctions = []
    x = np.arange(xspan[0], xspan[1], dx)  # grid for odeint
    for gamma in gammas:
        epsilon = 0.1
        for n in range(2):
            amplitude = 1e-3
            dAmplitude = 0.1
            for _ in range(100):
                dEpsilon = 0.2
                for _ in range(1000):
                    y0 = [amplitude, amplitude*np.sqrt(gamma + L**2 - epsilon)]  # initial condition
                    

                    sol = solve_ivp(shoot, xspan, y0, args=(epsilon, gamma), t_eval=x)  # solve ODE
                    ysol = sol.y
                    
                    if abs(ysol[1, -1] + np.sqrt(L**2 - epsilon) * ysol[0, -1]) < 1e-6:  # check convergence
                        break
                    
                    if (-1) ** (n+1) * (ysol[1, -1] + (np.sqrt(L**2 - epsilon) * ysol[0,-1])) < 0:
                        epsilon += dEpsilon
                    else:
                        epsilon -= dEpsilon
                        dEpsilon /= 2
                area = integrate.simpson(np.squeeze(ysol[0,:]), x=x, dx=.1)
                if area - 1 < 1e-6:
                    break
                if area < 1:
                    amplitude += dAmplitude
                elif area > 1:
                    amplitude -= dAmplitude/2
                    dAmplitude /= 2
            normedEigenfunction = ysol[0, :]/np.sqrt(np.trapezoid(ysol[0,:]**2, x))
            if gamma > 0:
                positiveGammaEigenfunctions.append(abs(normedEigenfunction))
                positiveGammaEpsilons.append(epsilon)
            else:
                negativeGammaEigenfunctions.append(abs(normedEigenfunction))
                negativeGammaEpsilons.append(epsilon)
            plt.plot(x, abs(normedEigenfunction))
            epsilon += .01

    A5 = np.matrix(np.transpose(positiveGammaEigenfunctions))
    A6 = np.array(positiveGammaEpsilons)

    A7 = np.matrix(np.transpose(negativeGammaEigenfunctions))
    A8 = np.array(negativeGammaEpsilons)

    plt.figure(1)
    plt.plot(x, A5)
    plt.legend(A6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Positive Gamma')
    plt.grid(True)

    plt.figure(2)
    plt.plot(x, A7)
    plt.legend(A8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Negative Gamma')
    plt.grid(True)

    plt.show()

    return A5, A6, A7, A8


# A3, A4 = partB()
A5, A6, A7, A8 = partC(2, .1, [0.05, -0.05])