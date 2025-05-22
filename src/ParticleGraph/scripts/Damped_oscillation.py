import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class Oscillator:
    def __init__(self, tau=10.0, freq=0.5):
        self.tau = tau
        self.freq = freq

    def get_derivative(self, t, y, input_val):
        gamma = 1 / self.tau
        omega = 2 * np.pi * self.freq
        dy = np.zeros(2)
        dy[0] = y[1]
        dy[1] = -2 * gamma * y[1] - omega**2 * y[0] + input_val
        return dy


osc = Oscillator()

def wrapped_derivative(t, y):
    input_val = 1.0  # Constant input to test system
    return osc.get_derivative(t, y, input_val)


t_span = (0, 100)
t_eval = np.linspace(*t_span, 1000)
y0 = [0.0, 0.0]

sol = solve_ivp(wrapped_derivative, t_span, y0, t_eval=t_eval, method='RK45')

plt.plot(sol.t, sol.y[0], label='y(t) - Oscillator Output')
plt.xlabel('Time')
plt.ylabel('y')
plt.title('Oscillator Response to Constant Input')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
