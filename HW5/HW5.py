from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.linalg import lu, solve_triangular, solve
from scipy.sparse import spdiags
from scipy.sparse.linalg import bicgstab, gmres
import time


def matrices(L, n):
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

    firstDegreeX = spdiags([e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()], [-(N-n),-n,n, (N-n)], N, N).toarray()/(2*dx)

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

    return secondDegreeXY, firstDegreeX, firstDegreeY

def secondXY_sparse_modified(L, n):
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

    center = np.copy(e1)
    center[0,0] /= 2 # ****REMOVE THIS FOR BASE MATRIX*****

    # Place diagonal elements
    diagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
                e2.flatten(), -4 * center.flatten(), e3.flatten(), 
                e4.flatten(), e1.flatten(), e1.flatten()]
    offsets = [-(N-n), -n, -n+1, -1, 0, 1, n-1, n, (N-n)]

    secondDegreeXY = spdiags(diagonals, offsets, N, N).toarray()/dx**2

    return secondDegreeXY

def partA():
    # Define parameters
    tspan = np.arange(0, 4.5, .5)
    nu = 0.001
    Lx, Ly = 20, 20
    nx, ny = 64, 64
    N = nx * ny

    #Generate derivative matrices
    secondXY, firstX, firstY = matrices(Lx/2, nx)

    # Define spatial domain and initial conditions
    x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
    x = x2[:nx]
    y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
    y = y2[:ny]
    X, Y = np.meshgrid(x, y)
    w = 1 * np.exp(0.05 * -Y**2 - X**2) # Initialize as complex

    # Define spectral k values
    kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
    kx[0] = 1e-6
    ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
    ky[0] = 1e-6
    KX, KY = np.meshgrid(kx, ky)
    K = KX**2 + KY**2

    # Define the ODE system
    def spc_rhs(t, w, nx, ny, N, K, nu, secondXY, firstX, firstY):
        wt = fft2(w.reshape((nx, ny)))
        
        psi = np.real(ifft2(-wt/K).reshape(N))  

        psix = firstX@psi
        psiy = firstY@psi
        wx = firstX@w
        wy = firstY@w

        diffusion = secondXY @ w
        advection = -psix*wy + wx*psiy


        rhs = (nu * diffusion + advection)
        return rhs
    
    # Solve the ODE and plot the results
    wt0 = w.reshape(N)
    start_time = time.time()
    sol = solve_ivp(spc_rhs, [0, 4], wt0, t_eval = tspan, method='RK45', args=(nx, ny, N, K, nu, secondXY, firstX, firstY))
    end_time = time.time()
    print(f"Elapsed time for FFT: {(end_time - start_time): .2f} seconds")
    wtsol = sol.y

    # for j, t in enumerate(tspan):
    #     w = wtsol[:N,j].reshape((nx, ny))
    #     plt.subplot(3, 3, j + 1)
    #     plt.pcolor(x, y, w, shading='interp')
    #     plt.title(f'Time: {t}')
    #     plt.colorbar()

    # plt.tight_layout()
    # plt.show()

    return wtsol

def partB():
    global residuals 
    residuals = np.array([])
    def callbackBICGSTAB(args):
        global residuals 
        if len(residuals) == 0:
            residuals = args
        else:
            residuals = np.vstack([residuals, args])
    
    def callbackGMRES(args):
        global residuals 
        residuals = np.append(residuals, args)

   # Define parameters
    tspan = np.arange(0, 4.5, .5)
    nu = 0.001
    Lx, Ly = 20, 20
    nx, ny = 64, 64
    N = nx * ny

    #Generate derivative matrices
    print("Generating matrices")
    secondXY, firstX, firstY = matrices(Lx/2, nx)
    secondXY[0,0] /= 2

    secondXY_sparse = secondXY_sparse_modified(Lx/2, nx)

    P, L, U = lu(secondXY)
    print("Complete")

    # Define spatial domain and initial conditions
    x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
    x = x2[:nx]
    y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
    y = y2[:ny]
    X, Y = np.meshgrid(x, y)
    w = 1 * np.exp(0.05 * -Y**2 - X**2) # Initialize as complex

    # Define the ODE system
    def spc_rhs(t, w, nu, P, L, U, secondXY, firstX, firstY, secondXY_sparse, flag):
        
        if flag == "A/b":
            psi = solve(secondXY, w)
        elif flag == "LU Decomposition":
            Pb = np.dot(P, w)
            y = solve_triangular(L, Pb, lower=True)
            psi = solve_triangular(U, y)
        elif flag == "BICGSTAB":
            psi, _ = (bicgstab(secondXY_sparse, w, atol = 1e-4, rtol = 1e-4, callback=callbackBICGSTAB))
        elif flag == "GMRES":
            psi, _ = (gmres(secondXY_sparse, w, atol=1e-4, rtol=1e-4, callback=callbackGMRES))
        else:
            raise Exception(f"Flag not recognized: {flag}")

        psix = firstX@psi
        psiy = firstY@psi
        wx = firstX@w
        wy = firstY@w

        diffusion = secondXY @ w
        advection = - psix*wy + wx*psiy


        rhs = (nu * diffusion + advection)
        return rhs
    
    # Solve the ODE and plot the results
    A2 = None
    A3 = None
    wt0 = w.reshape(N)
    
    residualsDict = {"BICGSTAB" : [],
                      "GMRES" : []}

    #keys: ['LU Decomposition', "A/b", "BICGSTAB", "GMRES"]
    for flag in ['LU Decomposition', "A/b", "BICGSTAB", "GMRES"]:
        print(f"Solving {flag}")
        start_time = time.time()
        sol = solve_ivp(spc_rhs, [0, 4], wt0, t_eval = tspan, method='RK45', args=(nu, P, L, U, secondXY, firstX, firstY, secondXY_sparse, flag))
        end_time = time.time()
        print(f"Elapsed time for {flag}: {(end_time - start_time): .2f} seconds")
        print(f"Residuals shape: {residuals.shape}")
        if len(residuals) != 0:
            residualsDict[flag] = residuals
            residuals = np.array([])
        wtsol = sol.y

        if flag == "LU Decomposition":
            A3 = wtsol
        elif flag == "A/b":
            A2 = wtsol
        else:
            for j, t in enumerate(tspan):
                w = wtsol[:N,j].reshape((nx, ny))
                plt.subplot(3, 3, j + 1)
                plt.pcolor(x, y, w, shading='auto')
                plt.title(f'Time: {t}')
                plt.colorbar()

            plt.tight_layout()
            plt.show()
            

    return A2, A3, residualsDict


def partC():
    # Define parameters
    tspan = np.arange(0, 45.5, .5)
    nu = 0.001
    Lx, Ly = 20, 20
    nx, ny = 128, 128
    N = nx * ny

    #Generate derivative matrices
    secondXY, firstX, firstY = matrices(Lx/2, nx)

    # Define spatial domain and initial conditions
    x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
    x = x2[:nx]
    y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
    y = y2[:ny]
    X, Y = np.meshgrid(x, y)
    w = 1 * np.exp(0.05 * -(Y+2)**2 - (X+3)**2) # Left pair
    w += 1 * np.exp(0.05 * -(Y-2)**2 - (X+3)**2)
    
    w -= 1 * np.exp(0.05 * -(Y+2)**2 - (X-3)**2) # Right pair
    w -= 1 * np.exp(0.05 * -(Y-2)**2 - (X-3)**2)

    # Define spectral k values
    kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
    kx[0] = 1e-6
    ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
    ky[0] = 1e-6
    KX, KY = np.meshgrid(kx, ky)
    K = KX**2 + KY**2

    # Define the ODE system
    def spc_rhs(t, w, nx, ny, N, K, nu, secondXY, firstX, firstY):
        wt = fft2(w.reshape((nx, ny)))
        
        psi = np.real(ifft2(-wt/K).reshape(N))  

        psix = firstX@psi
        psiy = firstY@psi
        wx = firstX@w
        wy = firstY@w

        diffusion = secondXY @ w
        advection = -psix*wy + wx*psiy


        rhs = (nu * diffusion + advection)
        return rhs
    
    # Solve the ODE and plot the results
    wt0 = w.reshape(N)
    start_time = time.time()
    sol = solve_ivp(spc_rhs, [0, 45], wt0, t_eval = tspan, method='RK45', args=(nx, ny, N, K, nu, secondXY, firstX, firstY))
    end_time = time.time()
    print(f"Elapsed time for FFT: {(end_time - start_time): .2f} seconds")
    wtsol = sol.y

    fig, ax = plt.subplots()
    def animation_update(frame):
        print(f"frame: {frame}")
        w = np.real(wtsol[:N, frame].reshape((nx, ny)))
        ax.pcolor(x, y, w, shading='auto')
        ax.title.set_text(f'Time: {round(tspan[frame],2)}')

    print(f"Total frames: {len(tspan)}")
    ani = FuncAnimation(fig, animation_update, frames=len(tspan), interval=500, repeat=False)
    ani.save("opposite_charge_collision.gif", dpi=300, writer=PillowWriter(fps=10))

    return wtsol


def partD():
    # Define parameters
    tspan = np.arange(0, 30.5, .5)
    nu = 0.001
    Lx, Ly = 20, 20
    nx, ny = 128, 128
    N = nx * ny

    #Generate derivative matrices
    secondXY, firstX, firstY = matrices(Lx/2, nx)

    # Define spatial domain and initial conditions
    x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
    x = x2[:nx]
    y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
    y = y2[:ny]
    X, Y = np.meshgrid(x, y)
    w = 1 * np.exp(0.05 * -Y**2 - X**2) # Initialize as complex

    # Define spectral k values
    kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
    kx[0] = 1e-6
    ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
    ky[0] = 1e-6
    KX, KY = np.meshgrid(kx, ky)
    K = KX**2 + KY**2

    # Define the ODE system
    def spc_rhs(t, w, nx, ny, N, K, nu, secondXY, firstX, firstY):
        wt = fft2(w.reshape((nx, ny)))

        psi = np.real(ifft2(-wt/K).reshape(N))  

        psix = firstX@psi
        psiy = firstY@psi
        wx = firstX@w
        wy = firstY@w

        diffusion = secondXY @ w
        advection = -psix*wy + wx*psiy


        rhs = (nu * diffusion + advection)
        return rhs
    
    # Solve the ODE and plot the results
    wt0 = w.reshape(N)
    sol = solve_ivp(spc_rhs, [0, 30], wt0, t_eval = tspan, method='RK45', args=(nx, ny, N, K, nu, secondXY, firstX, firstY))
    wtsol = sol.y

    fig, ax = plt.subplots()
    def animation_update(frame):
        print(f"frame: {frame}")
        w = np.real(wtsol[:N, frame].reshape((nx, ny)))
        ax.pcolor(x, y, w, shading='auto')
        ax.title.set_text(f'Time: {round(tspan[frame],2)}')

    print(f"Total frames: {len(tspan)}")
    ani = FuncAnimation(fig, animation_update, frames=len(tspan), interval=500, repeat=False)
    ani.save("partDMovie.gif", dpi=300, writer=PillowWriter(fps=10))


# A1 = partA()

A2, A3, residualsDict = partB()


# partD()

# partC()
