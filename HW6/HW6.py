import numpy as np
from scipy.integrate import solve_ivp
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def chebychev(N):
    if N == 0:
        D = 0
        x = 1
    else:
        n = np.arange(0, N + 1)
        x = np.cos(np.pi * n / N).reshape(N + 1, 1)
        c = (np.hstack(( [2.], np.ones(N - 1), [2.])) * (-1)**n).reshape(N + 1, 1)
        X = np.tile(x, (1, N + 1))
        dX = X - X.T
        D = np.dot(c, 1./c.T) / (dX + np.eye(N + 1))
        D -= np.diag(np.sum(D.T, axis= 0))
    return D, x.reshape(N + 1)

def partA():
    # Define parameters
    tspan = np.arange(0, 4.5, .5)
    Lx, Ly = 20, 20
    nx, ny = 64, 64
    N = nx * ny
    beta = 1
    D1, D2 = 0.1, 0.1
    m = 1

    # Define spatial domain
    x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
    x = x2[:nx]
    y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
    y = y2[:ny]
    X, Y = np.meshgrid(x, y)

    # Define initial conditions
    r = np.sqrt(X**2 + Y**2)
    theta = np.angle(X + 1j * Y)
    u = np.tanh(r) * np.cos(m * theta - r)
    v = np.tanh(r) * np.sin(m * theta - r)

    uFFT = fft2(u).flatten()
    vFFT = fft2(v).flatten()
    y0 = np.hstack((uFFT, vFFT))

    # Define spectral k values
    kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
    kx[0] = 1e-6
    ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
    ky[0] = 1e-6
    KX, KY = np.meshgrid(kx, ky)
    K = KX**2 + KY**2

    def diffusion_rhs(t, y, beta, D1, D2, nx, ny, N):
        uFFT = y[:N]
        vFFT = y[N:]

        # Reshape u and v
        uFFT = uFFT.reshape((nx, ny))
        vFFT = vFFT.reshape((nx, ny))
        u_t = ifft2(uFFT)
        v_t = ifft2(vFFT)

        # Compute Laplacian
        uLaplacian = K*uFFT
        vLaplacian = K*vFFT

        # Define A, lamda function, and omega function
        Asquared = (u_t**2 + v_t**2)**2
        lam = 1 - Asquared
        omega = -beta * Asquared

        # Compute time derivatives
        uRHS = fft2(lam * u_t - omega * v_t) - D1 * uLaplacian
        vRHS = fft2(omega * u_t + lam * v_t) - D2 * vLaplacian
        uRHS = uRHS.reshape(N)
        vRHS = vRHS.reshape(N)

        return np.hstack((uRHS, vRHS))
    
    params = (beta, D1, D2, nx, ny, N)
    sol = solve_ivp(diffusion_rhs, [tspan[0], tspan[-1]], y0, t_eval=tspan, method="RK45", args=params)
    uvsol = sol.y
    print(uvsol.shape)
    print(f"Status: {sol.status}")
    print(f"Message: {sol.message}")
    print(f"first val: {uvsol[0,0]}")


    if (sol.status == -1):
        exit(0)

    fig, ax = plt.subplots()
    def animation_update(frame):
        print(f"frame: {frame}")
        w = np.real(ifft2(uvsol[:N, frame].reshape((nx, ny))))
        ax.pcolor(x, y, w, shading='auto')
        ax.title.set_text(f'Time: {round(tspan[frame],2)}')

    print(f"Total frames: {len(tspan)}")
    ani = FuncAnimation(fig, animation_update, frames=len(tspan), interval=500, repeat=False)
    ani.save("part_a_movie.gif", dpi=300, writer=PillowWriter(fps=10))

    return uvsol

def partB():
    N = 30
    D, x = chebychev(N)

A1 = partA()