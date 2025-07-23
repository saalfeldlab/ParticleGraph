#%%
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
import myspectral_funcs
from importlib import reload


save_path = "./figs/"

# --------------------------
# 1. Set up the time grid
# --------------------------
T = 50.0            # Total time duration
fps = 10
N = int(T*fps)            # Number of time points
t = np.linspace(0, T, N)  # Time vector from 0 to T

# --------------------------
# 2. Define main GP kernels
# --------------------------
# Low- and high-frequency oscillation parameters
f_low = 0.5         # Low-frequency (Hz)
f_high = 5.0        # High-frequency (Hz)
ell_low = 1.0       # Lengthscale for low-frequency kernel
ell_high = 0.2      # Lengthscale for high-frequency kernel
sigma_low = 1.0     # Amplitude for low-frequency kernel
sigma_high = 0.5    # Amplitude for high-frequency kernel

def oscillatory_kernel(t1, t2, f, ell, sigma):
    """
    Oscillatory Gaussian kernel:
      k(t1, t2) = sigma^2 * exp(-(t1-t2)^2/(2*ell^2)) * cos(2*pi*f*(t1-t2))
    Combines smooth Gaussian decay with cosine oscillation at frequency f.
    """
    dt = t1 - t2
    return sigma**2 * np.exp(-0.5 * (dt/ell)**2) * np.cos(2 * np.pi * f * dt)

# Precompute stationary covariance matrices for low and high components
K_low  = np.array([[oscillatory_kernel(t[i], t[j], f_low,  ell_low,  sigma_low)  for j in range(N)] for i in range(N)])
K_high = np.array([[oscillatory_kernel(t[i], t[j], f_high, ell_high, sigma_high) for j in range(N)] for i in range(N)])

# --------------------------
# 3. Build envelope processes
# --------------------------
# Very-low-frequency GP parameters for envelopes
f_env    = 0.001      # Envelope oscillation frequency (Hz)
ell_env  = 3.0      # Envelope lengthscale
sigma_env= 1.0      # Envelope variance

# Build covariance for the envelope GP
K_env = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        K_env[i, j] = oscillatory_kernel(t[i], t[j], f_env, ell_env, sigma_env)
K_env += 1e-6 * np.eye(N)  # Add small nugget for numerical stability

# Sample two independent envelopes for low- and high-frequency bands
e_low  = np.linspace(-10, 10, N)
e_high  = np.random.multivariate_normal(np.zeros(N), K_env)
envelope_f_low = 0.01
envelope_f_high = 0.03
# e_low = np.sin(2*np.pi*envelope_f_low*t)
e_high = np.sin(2*np.pi*envelope_f_high*t)*10
# e_high = np.ones(N)
# e_high = np.random.multivariate_normal(np.zeros(N), K_env)

# Convert raw GP draws to (0,1) via a sigmoid so they can serve as weights
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

w_low  = sigmoid(e_low)   # Time-varying weight for the low-frequency kernel
w_high = sigmoid(e_high)  # Time-varying weight for the high-frequency kernel


# --------------------------
# 4. Build nonstationary covariance
# --------------------------
K_ns = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        # Each kernel is modulated by the product of its envelope values
        K_ns[i, j] = w_low[i]*w_low[j]*K_low[i, j] + w_high[i]*w_high[j]*K_high[i, j]
K_ns += 1e-6 * np.eye(N)  # Final nugget for stability

# --------------------------
# 5. Sample from the nonstationary GP
# --------------------------
y_ns = np.random.multivariate_normal(np.zeros(N), K_ns)

#%%

reload(myspectral_funcs)
xnt = y_ns[None]
ynt = None
fs = fps
window = "hann"
nperseg = int(fs*4)
noverlap = None
nfft = None
detrend = "constant"
return_onesided = True
scaling = "spectrum"
abs = True
return_coefs = True

pxy, freqs, coefs_xnkf = myspectral_funcs.estimate_spectrum(xnt, ynt =ynt, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, return_onesided=return_onesided, scaling=scaling, abs=abs, return_coefs=return_coefs)
coefs_xnkf = coefs_xnkf.compute()
pxy = pxy.compute()

#%%
# --------------------------
# 6. Plot envelopes and GP realization
# --------------------------
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Plot envelope processes
ax[0].plot(t, w_low,  label='Low-Freq Envelope')
ax[0].plot(t, w_high, label='High-Freq Envelope')
ax[0].set_ylabel('Envelope Weight')
ax[0].legend(loc='upper right')

# Plot one sample from the nonstationary GP
ax[1].plot(t, y_ns, color='tab:orange')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('GP Value')
ax[1].set_title('Nonstationary GP Sample with Random Envelopes')

plt.tight_layout()
plt.savefig(save_path + "gp_realization.png")
plt.close()

plt.figure()
plt.title("PSD")
plt.plot(freqs, pxy[0,0])
plt.savefig(save_path + "psd.png")
plt.close()

min_freq = freqs[0]
max_freq = freqs[-1]
spectrogram = np.abs(coefs_xnkf[0]).T[::-1]
xax = np.linspace(0, T, spectrogram.shape[1])
plt.figure()
cmap = plt.get_cmap("coolwarm")
plt.matshow(spectrogram, extent = [xax[0], xax[-1], min_freq, max_freq], cmap=cmap, aspect="auto")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar()
plt.savefig(save_path + "spectrogram.png")
plt.close()

#%%
reload(myspectral_funcs)
num_levels = 10
reps = 5
reload(myspectral_funcs)
coefs_xnkfs, freqs_ = myspectral_funcs.compute_multiscale_spectral_coefs(xnt=y_ns[None], fs=fps, window="hann", noverlap=None, detrend="constant", return_onesided=True, scaling="spectrum", axis=0, num_levels = num_levels, reps = reps)

spectrogram, final_freqs, min_freq, max_freq = myspectral_funcs.wrangle_multiscale_coefs(coefs_xnkfs, freqs_)

xax = np.linspace(0, T, spectrogram.shape[1])

plt.figure(figsize=(4, 3))
cmap = plt.get_cmap("coolwarm")
dxx = spectrogram.shape[1]
dxy = spectrogram.shape[0]
plt.matshow(spectrogram, extent = [xax[0], xax[-1], min_freq, max_freq], cmap=cmap, aspect=dxx/dxy)
ds = 3
plt.yticks(np.linspace(0, final_freqs[-1], len(final_freqs))[::ds], np.round(final_freqs[::ds], 2))
# plt.xticks(np.arange(len(xax)), xax)
plt.colorbar()
plt.title("Multiscale Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.savefig(save_path + "multiscale_spectrogram.png")
plt.close()




#%%

#%%
# reload(myspectral_funcs)
# N = len(t)
# freqs = myspectral_funcs.get_freqs(nperseg=N, fs=fps)[0]

# # lower frequency one can infer is a cycle of the full length of the signal
# lowest_freq = 1/(N/fps)
# highest_freq = (N/2)/(N/fps) # i.e.: 2/fps
# num_levels = 10
# freqs_ = (np.arange(0, int(N/2))/(N/fps))[::(int(N/2)//num_levels)]
# freqs_

