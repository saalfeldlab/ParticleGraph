import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, impulse






import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, lsim

# Circuit parameters
L = 1e-3     # 1 mH
C = 1e-6     # 1 uF
R = 10       # 10 Ohms

# Transfer function: H(s) = 1 / (LC s^2 + RC s + 1)
num = [1]
den = [L*C, R*C, 1]
system = TransferFunction(num, den)

# Time vector
t = np.linspace(0, 0.02, 2000)  # 0 to 20 ms

# Gaussian input pulse centered at 5 ms with width sigma=1 ms
t0 = 0.005
sigma = 0.001
u = np.exp(-((t - t0)**2) / (2 * sigma**2))

# Simulate system response to Gaussian input
t_out, y_out, _ = lsim(system, U=u, T=t)

# Plot input and output
plt.figure(figsize=(12, 5))

plt.plot(t*1000, u, label='Input Gaussian Pulse', color='orange')
plt.plot(t_out*1000, y_out, label='Output Response', color='blue')

plt.title("RLC Circuit Response to Gaussian Input Pulse")
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()










# New RLC parameters for higher frequency
L = 0.0001    # Lower inductance => higher frequency
C = 0.00001   # Lower capacitance => higher frequency
R = 1.0      # Lower resistance => less damping

# Transfer function: H(s) = 1 / (L*C*s^2 + R*C*s + 1)
num = [1]
den = [L*C, R*C, 1]

# Create transfer function system
system = TransferFunction(num, den)

# Time vector needs to be shorter to see rapid oscillations
t, y = impulse(system, T=np.linspace(0, 0.01, 1000))  # 0 to 10ms

# Impulse input for visualization (approximated)
impulse_input = np.zeros_like(t)
impulse_input[0] = 1 / (t[1] - t[0])

# Plot
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(t, impulse_input, color='orange')
plt.title("Impulse Input (approximated)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, y, color='blue')
plt.title("Impulse Response of RLC Circuit (Higher Frequency)")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.grid(True)

plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt

# RLC parameters
L = 0.001    # Henry
C = 0.0001   # Farad
R = 1.0      # Ohms

# Frequency range (log scale for better resolution)
frequencies = np.logspace(2, 6, 1000)  # from 100 Hz to 1 MHz
omega = 2 * np.pi * frequencies        # angular frequency

# Compute transfer function magnitude |H(jω)|
numerator = 1
denominator = 1 - (L * C) * omega**2 + 1j * (R * C * omega)
H_jw = numerator / denominator
H_mag = np.abs(H_jw)

# Plot magnitude response (dB optional)
plt.figure(figsize=(10, 6))
plt.semilogx(frequencies, 20 * np.log10(H_mag), label='|H(jω)| in dB')
plt.title("Frequency Response of Series RLC Circuit (Output Across Capacitor)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True, which='both', ls='--')
plt.axvline(1 / (2 * np.pi * np.sqrt(L * C)), color='red', linestyle='--', label='Resonant Frequency')
plt.legend()
plt.tight_layout()
plt.show()
