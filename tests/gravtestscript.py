"""
Extended Physical Models: Mutual Gravitational Interaction Simulation
-----------------------------------------------------------------------
This script simulates the motion of the Sun, Earth, and Mars under mutual gravitational
interactions using Newtonian mechanics. The ODE system is integrated using SciPy's solve_ivp.
Author: [Your Name]
Date: February 2025

Requirements:
- NumPy
- Matplotlib
- SciPy
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import datetime

# Define physical constants.
G = 6.67430e-11  # gravitational constant
AU = 1.496e11    # astronomical unit in meters
SECONDS_PER_DAY = 86400

# Masses of the bodies (in kg)
mass_sun = 1.98847e30
mass_earth = 5.972e24
mass_mars = 6.39e23

# Define initial conditions:
# Positions for Sun, Earth, and Mars (in meters).
x0 = [0, 0, 0,           # Sun position
      1 * AU, 0, 0,      # Earth position
      1.524 * AU, 0, 0]  # Mars position

# Velocities for Sun, Earth, and Mars (in m/s).
# Now each body has a full 3D velocity vector.
vx0 = [0, 0, 0,           # Sun velocity
       0, 29780, 0,       # Earth velocity ~29.78 km/s
       0, 24007, 0]       # Mars velocity ~24.01 km/s

# Combine into a single state vector (positions then velocities).
y0 = np.array(x0 + vx0, dtype=float)

masses = np.array([mass_sun, mass_earth, mass_mars])
n = len(masses)  # Number of bodies (3)

def n_body_equations(t, y):
    """Compute derivatives for an N-body gravitational system."""
    positions = y[:3*n].reshape((n, 3))
    velocities = y[3*n:].reshape((n, 3))
    accelerations = np.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i != j:
                r_vec = positions[j] - positions[i]
                r_mag = np.linalg.norm(r_vec)
                accelerations[i] += G * masses[j] * r_vec / r_mag**3
    dydt = np.concatenate([velocities.flatten(), accelerations.flatten()])
    return dydt

# Simulation time span.
t0 = 0
t_end = 365 * SECONDS_PER_DAY * 2  # 2 years
t_eval = np.linspace(t0, t_end, 1000)

# Integrate the ODE.
sol = solve_ivp(n_body_equations, (t0, t_end), y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

# Extract trajectories.
positions = sol.y[:3*n].reshape((n, 3, -1))
times = sol.t

# Plot trajectories in 3D.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = ['yellow', 'blue', 'red']
labels = ['Sun', 'Earth', 'Mars']

for i in range(n):
    ax.plot(positions[i, 0, :], positions[i, 1, :], positions[i, 2, :],
            color=colors[i], label=labels[i])
    ax.scatter(positions[i, 0, -1], positions[i, 1, -1], positions[i, 2, -1],
               color=colors[i], s=50)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Extended Physical Model: Mutual Gravitational Interaction")
ax.legend()
plt.show()
