import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy import integrate
from scipy.sparse.linalg import eigs
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

def partA():
    def shoot(x, y, epsilon):
        return [y[1], y[0]*(x**2 - epsilon)]

    xspan = [-4, 4.1]  # x range
    # initial derivative value
    # step size for derivative adjustment
    epsilon = 0
    epsilons = []
    eigenfunctions = []

    for n in range(5):
        
        dEpsilon = 0.1
        for j in range(1000):
            y0 = [1, np.sqrt(16 - epsilon)]  # initial condition
            x = np.arange(xspan[0], xspan[1], .1)  # grid for odeint
            sol = solve_ivp(shoot, xspan, y0, args=(epsilon,), t_eval=x)  # solve ODE
            ysol = sol.y
            
            if abs(ysol[1, -1] + np.sqrt(16 - epsilon) * ysol[0, -1]) < 1e-4:  # check convergence
                break
            
            if (-1) ** (n+1) * (ysol[1, -1] + (np.sqrt(16 - epsilon) * ysol[0,-1])) < 0:
                epsilon += dEpsilon
            else:
                epsilon -= dEpsilon
                dEpsilon /= 2
        normedEigenfunction = ysol[0, :]/np.sqrt(np.trapz(ysol[0,:]**2, x))
        eigenfunctions.append(abs(normedEigenfunction))
        epsilons.append(epsilon)
        epsilon += 1

    A1 = np.array(np.transpose(eigenfunctions))
    A2 = np.array(epsilons)

    return A1, A2

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
    
    A[0,0] += 4/3/(-dx**2)
    A[0,1] += -1/3/(-dx**2)
    A[-1,-1] += 4/3/(-dx**2)
    A[-1,-2] += -1/3/(-dx**2)
    
    D, V = eigs(A, k=5, which='SR')
    
    D5 = D
    V5 = V

    A3 = [] #eigenfunctions
    A4 = np.asarray(D5, float) #eigenvalues

    for k in range(5):
        arrEigenvector = np.squeeze(np.asarray(V5[:,k], float))

        leftBoundary = 4/3 * arrEigenvector[0] - 1/3 * arrEigenvector[1]
        rightBoundary = 4/3 * arrEigenvector[-1] - 1/3 * arrEigenvector[-2]
        arrEigenvector = np.array([leftBoundary, *arrEigenvector, rightBoundary])

        normedEigenfunction = arrEigenvector/np.sqrt(np.trapezoid(arrEigenvector**2, dx=dx))
        plt.plot(x, abs(normedEigenfunction))
        A3.append(abs(normedEigenfunction))

    A3 = np.array(A3).T

    plt.legend(np.asarray(D5, float))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solution Plot')
    plt.grid(True)
    # plt.show()

    return A3, A4


def partC(L, dx, gammas):
    def shoot(x, y, epsilon, gamma):
        return [y[1], y[0]*(gamma*abs(y[0])**2 + x**2 - epsilon)]

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
        for n in range(2): #find first 2 eigenfunctions/values
            amplitude = 1e-3
            dAmplitude = 0.1
            for _ in range(100): #iterating on amplitude
                dEpsilon = 0.2
                for _ in range(1000): #iterating on epsilon
                    y0 = [amplitude, amplitude*np.sqrt(L**2 - epsilon)]  # initial condition

                    sol = solve_ivp(shoot, xspan, y0, args=(epsilon, gamma), t_eval=x)  # solve ODE
                    ysol = sol.y
                    
                    if abs(ysol[1, -1] + np.sqrt(L**2 - epsilon) * ysol[0, -1]) < 1e-6:  # check convergence
                        break
                    
                    if (-1) ** (n+1) * (ysol[1, -1] + (np.sqrt(L**2 - epsilon) * ysol[0,-1])) < 0:
                        epsilon += dEpsilon
                    else:
                        epsilon -= dEpsilon
                        dEpsilon /= 2
                area = integrate.simpson(ysol[0,:], x=x, dx=.1)
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
            epsilon += .1

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

    # plt.show()

    return A5, A6, A7, A8

def partC(L, dx, gammas):
    def shoot(x, y, epsilon, gamma):
        return [y[1], y[0]*(gamma*abs(y[0])**2 + x**2 - epsilon)]

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
        for n in range(2): #find first 2 eigenfunctions/values
            amplitude = 1e-3
            dAmplitude = 0.1
            for _ in range(100): #iterating on amplitude
                dEpsilon = 0.2
                for _ in range(1000): #iterating on epsilon
                    y0 = [amplitude, amplitude*np.sqrt(L**2 - epsilon)]  # initial condition

                    sol = solve_ivp(shoot, xspan, y0, args=(epsilon, gamma), t_eval=x)  # solve ODE
                    ysol = sol.y
                    
                    if abs(ysol[1, -1] + np.sqrt(L**2 - epsilon) * ysol[0, -1]) < 1e-4:  # check convergence
                        break
                    
                    if (-1) ** (n+1) * (ysol[1, -1] + (np.sqrt(L**2 - epsilon) * ysol[0,-1])) < 0:
                        epsilon += dEpsilon
                    else:
                        epsilon -= dEpsilon
                        dEpsilon /= 2
                area = np.trapezoid(ysol[0,:], x=x, dx=.1)
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
            epsilon += .1

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

    # plt.show()

    return A5, A6, A7, A8

def partD(L, tolerances):
    def shoot(x, y, epsilon):
        return [y[1], y[0]*(x**2 - epsilon)]
    E = 1
    y0 = [1, np.sqrt(L**2 - E)]
    xspan = [-L,L]

    rk45StepSizes = []
    rk23StepSizes = []
    radauStepSizes = []
    bdfStepSizes = []

    for TOL in tolerances:
        options = {'rtol': TOL, 'atol': TOL}
        rk45sol = solve_ivp(shoot, xspan, y0, method='RK45', args=(E,), **options)
        rk45T = rk45sol.t
        rk45StepSize = np.mean(np.diff(rk45T))
        rk45StepSizes.append(rk45StepSize)

        rk23sol = solve_ivp(shoot, xspan, y0, method='RK23', args=(E,), **options)
        rk23T = rk23sol.t
        rk23StepSize = np.mean(np.diff(rk23T))
        rk23StepSizes.append(rk23StepSize)

        radausol = solve_ivp(shoot, xspan, y0, method='Radau', args=(E,), **options)
        radauT = radausol.t
        radauStepSize = np.mean(np.diff(radauT))
        radauStepSizes.append(radauStepSize)

        bdfsol = solve_ivp(shoot, xspan, y0, method='BDF', args=(E,), **options)
        bdfT = bdfsol.t
        bdfStepSize = np.mean(np.diff(bdfT))
        bdfStepSizes.append(bdfStepSize)
        
    plt.plot(rk45StepSizes, tolerances)
    plt.plot(rk23StepSizes, tolerances)
    plt.plot(radauStepSizes, tolerances)
    plt.plot(bdfStepSizes, tolerances)
    plt.yscale('log')
    plt.xscale('log')
    # plt.show()

    rk45StepSizes = np.log10(rk45StepSizes)
    rk23StepSizes = np.log10(rk23StepSizes)
    radauStepSizes = np.log10(radauStepSizes)
    bdfStepSizes = np.log10(bdfStepSizes)

    tolerances = np.log10(tolerances)

    rk45Slope = np.polyfit(tolerances, rk45StepSizes, 1)[0]
    rk23Slope = np.polyfit(tolerances, rk23StepSizes, 1)[0]
    radauSlope = np.polyfit(tolerances, radauStepSizes, 1)[0]
    bdfSlope = np.polyfit(tolerances, bdfStepSizes, 1)[0]

    A9 = np.array([rk45Slope, rk23Slope, radauSlope, bdfSlope])
    return A9
    

def partE(A1, A2, A3, A4):
    def expon(x):
        return np.exp(-1/2* x**2)
    xspan = np.arange(-4, 4.1, .1)
    exact_vecs = np.zeros((81, 81))

    exact_vecs[:,0] = np.array((np.pi ** (-1 / 4)) * expon(xspan)).T
    exact_vecs[:,1] = np.array(np.sqrt(2) * (np.pi ** (-1 / 4)) * xspan * expon(xspan)).T
    exact_vecs[:,2] = np.array(1 / (np.sqrt(2) * (np.pi ** (-1 / 4))) * (2 * (xspan ** 2) - 1) * expon(xspan)).T
    exact_vecs[:,3] = np.array(1 / (np.sqrt(3) * (np.pi ** (-1 / 4))) * (2 * (xspan ** 3) - 3 * xspan) * expon(xspan)).T
    exact_vecs[:,4] = np.array(1 / (2 * np.sqrt(6) * (np.pi ** (-1 / 4))) * (4 * (xspan ** 4) - 12 * (xspan ** 2) + 3) * expon(xspan)).T

    partADifferenceFunctionNorms = []
    partBDifferenceFunctionNorms = []
    partADifferenceValues = []
    partBDifferenceValues = []

    for col in range(5):
        partAFunctionDiff = A1[:,col] - abs(exact_vecs[:,col])
        partADifferenceFunctionNorms.append(np.trapezoid(partAFunctionDiff**2, xspan))

        partAValueDiff = 100 * abs(A2[col] - (2*col + 1))/(2*col + 1)
        partADifferenceValues.append(partAValueDiff)

        partBFunctionDiff = A3[:,col] - abs(exact_vecs[:,col])
        partBDifferenceFunctionNorms.append(np.trapezoid(partBFunctionDiff**2, xspan))

        partBValueDiff = 100 * abs(A4[col] - (2*col + 1))/(2*col + 1)
        partBDifferenceValues.append(partBValueDiff)

    return partADifferenceFunctionNorms, partADifferenceValues, partBDifferenceFunctionNorms, partBDifferenceValues





A1, A2 = partA()
A3, A4 = partB()
A5, A6, A7, A8 = partC(2, .1, [0.05, -0.05])
tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
A9 = partD(2, tols)
A10, A11, A12, A13 = partE(A1, A2, A3, A4)
