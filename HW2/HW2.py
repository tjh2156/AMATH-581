import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def shoot(y, x, epsilon):
    return [y[1], y[0]*(x**2 - epsilon)]

xspan = [-4, 4]  # x range
 # initial derivative value
  # step size for derivative adjustment
epsilon = 0
epsilons = []
eigenfunctions = []

for n in range(5):
    
    dEpsilon = 0.1
    for j in range(1000):
        y0 = [1, np.sqrt(16 - epsilon)]  # initial condition
        x = np.arange(xspan[0], xspan[1] + .1, .1)  # grid for odeint
        ysol = odeint(shoot, y0, x, (epsilon,))  # solve ODE
        
        if abs(ysol[-1, 1] + np.sqrt(16 - epsilon) * ysol[-1, 0]) < 1e-6:  # check convergence
            break
        
        if (-1) ** (n+1) * (ysol[-1, 1] + (np.sqrt(16 - epsilon) * ysol[-1,0])) < 0:
            epsilon += dEpsilon
        else:
            epsilon -= dEpsilon
            dEpsilon /= 2
    normedEigenfunction = ysol[:, 0]/np.sqrt(np.trapz(ysol[:,0]**2, x))
    eigenfunctions.append(abs(normedEigenfunction))
    epsilons.append(epsilon)
    plt.plot(x, normedEigenfunction)
    epsilon += 1

A1 = np.matrix(np.transpose(eigenfunctions))
A2 = np.array(epsilons) 
print(A1.shape)

plt.legend(epsilons)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution Plot')
plt.grid(True)
plt.show()

# Plot the solution
