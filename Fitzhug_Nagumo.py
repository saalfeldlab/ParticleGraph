import numpy as np
import matplotlib.pyplot as plt



# Parameters
a = 0.7
b = 0.8
epsilon = 0.18

# Simulation settings
T = 400       # total time
dt = 0.1      # time step
n_steps = int(T / dt)
time = np.linspace(0, T, n_steps)

# Initialize variables
v = np.zeros(n_steps)
w = np.zeros(n_steps)
I_ext = np.zeros(n_steps)

# Initial conditions
v[0] = -1.0
w[0] = 1.0

# External excitation: periodic pulse every 30s lasting 1s
pulse_interval = 80.0  # seconds
pulse_duration = 1.0   # seconds
pulse_amplitude = 0.8  # strength of excitation

for i, t in enumerate(time):
    if (t % pulse_interval) < pulse_duration:
        I_ext[i] = pulse_amplitude

# Rollout using Euler method
for i in range(n_steps - 1):
    dv = v[i] - (v[i]**3)/3 - w[i] + I_ext[i]
    dw = epsilon * (v[i] + a - b * w[i])
    v[i+1] = v[i] + dt * dv
    w[i+1] = w[i] + dt * dw

# Plotting
plt.figure(figsize=(12, 6))

# Time series
plt.subplot(2, 1, 1)
plt.plot(time, v, label='v (membrane potential)')
plt.plot(time, w, label='w (recovery variable)', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('State')
plt.title('FitzHugh-Nagumo with Periodic Excitation')
plt.legend()
plt.grid(True)

# Plot I_ext
plt.subplot(2, 1, 2)
plt.plot(time, I_ext, color='red', label='External input $I_{ext}$')
plt.xlabel('Time')
plt.ylabel('Input current')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 8))
plt.plot(v, w)
plt.xlabel('v')
plt.ylabel('w')
plt.title('Phase Portrait')
plt.grid(True)
plt.tight_layout()
plt.show()
