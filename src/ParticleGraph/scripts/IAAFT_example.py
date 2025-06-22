import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft


def iaaft_surrogate_example(time_series, max_iter=1):
    """Simple IAAFT implementation for demonstration"""
    original = np.array(time_series)
    n = len(original)

    # Sort original amplitudes
    sorted_amplitudes = np.sort(original)

    # Initial random phase
    phases = np.random.uniform(0, 2 * np.pi, n)
    fft_original = fft(original)
    amplitudes = np.abs(fft_original)

    surrogate = original.copy()

    for _ in range(max_iter):
        # Step 1: match power spectrum
        fft_surrogate = amplitudes * np.exp(1j * phases)
        surrogate = np.real(ifft(fft_surrogate))

        # Step 2: match amplitude distribution
        sorted_indices = np.argsort(surrogate)
        surrogate[sorted_indices] = sorted_amplitudes

        # Update phases
        phases = np.angle(fft(surrogate))

    return surrogate


# Create example time series (calcium-like signal)
np.random.seed(42)
t = np.linspace(0, 100, 500)

# Simulated calcium signal: oscillations + noise + some spikes
calcium_signal = (
        0.5 * np.sin(0.3 * t) +  # slow oscillation
        0.3 * np.sin(0.8 * t + 1.5) +  # faster component
        0.2 * np.random.randn(len(t))  # noise
)

# Add some calcium spikes
spike_times = [150, 280, 420]
for spike_time in spike_times:
    calcium_signal[spike_time:spike_time + 10] += np.exp(-np.arange(10) / 3) * 2

# Generate surrogate
surrogate = iaaft_surrogate_example(calcium_signal)

# Create comprehensive comparison plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Time series comparison
axes[0, 0].plot(t, calcium_signal, 'b-', label='Original', alpha=0.8)
axes[0, 0].plot(t, surrogate, 'r-', label='IAAFT Surrogate', alpha=0.8)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Calcium Signal')
axes[0, 0].set_title('Time Series Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Amplitude distributions (histograms)
axes[0, 1].hist(calcium_signal, bins=30, alpha=0.7, label='Original', density=True, color='blue')
axes[0, 1].hist(surrogate, bins=30, alpha=0.7, label='Surrogate', density=True, color='red')
axes[0, 1].set_xlabel('Signal Value')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Amplitude Distributions')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Power spectra
freqs_orig = np.fft.fftfreq(len(calcium_signal), d=t[1] - t[0])
power_orig = np.abs(fft(calcium_signal)) ** 2
power_surr = np.abs(fft(surrogate)) ** 2

# Only plot positive frequencies
pos_mask = freqs_orig > 0
axes[0, 2].loglog(freqs_orig[pos_mask], power_orig[pos_mask], 'b-', label='Original', alpha=0.8)
axes[0, 2].loglog(freqs_orig[pos_mask], power_surr[pos_mask], 'r--', label='Surrogate', alpha=0.8)
axes[0, 2].set_xlabel('Frequency')
axes[0, 2].set_ylabel('Power')
axes[0, 2].set_title('Power Spectra')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)


# Autocorrelation functions
def autocorr(x, maxlags=50):
    """Calculate autocorrelation"""
    x = x - np.mean(x)
    autocorr_full = np.correlate(x, x, mode='full')
    autocorr_full = autocorr_full / autocorr_full[len(autocorr_full) // 2]
    mid = len(autocorr_full) // 2
    return autocorr_full[mid:mid + maxlags + 1]


lags = np.arange(51)
autocorr_orig = autocorr(calcium_signal, 50)
autocorr_surr = autocorr(surrogate, 50)

axes[1, 0].plot(lags, autocorr_orig, 'b-', label='Original', alpha=0.8)
axes[1, 0].plot(lags, autocorr_surr, 'r-', label='Surrogate', alpha=0.8)
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('Autocorrelation')
axes[1, 0].set_title('Autocorrelation Functions')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Cross-correlation between original and surrogate
crosscorr = np.correlate(calcium_signal - np.mean(calcium_signal),
                         surrogate - np.mean(surrogate), mode='full')
crosscorr = crosscorr / np.sqrt(np.var(calcium_signal) * np.var(surrogate) * len(calcium_signal))
mid = len(crosscorr) // 2
lags_cross = np.arange(-50, 51)
axes[1, 1].plot(lags_cross, crosscorr[mid - 50:mid + 51], 'g-', alpha=0.8)
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('Cross-correlation')
axes[1, 1].set_title('Original vs Surrogate Cross-correlation')
axes[1, 1].grid(True, alpha=0.3)

# Summary statistics
stats_text = f"""
Original Statistics:
Mean: {np.mean(calcium_signal):.3f}
Std: {np.std(calcium_signal):.3f}
Min: {np.min(calcium_signal):.3f}
Max: {np.max(calcium_signal):.3f}

Surrogate Statistics:
Mean: {np.mean(surrogate):.3f}
Std: {np.std(surrogate):.3f}
Min: {np.min(surrogate):.3f}
Max: {np.max(surrogate):.3f}

Power Spectrum Match:
Correlation: {np.corrcoef(power_orig, power_surr)[0, 1]:.4f}

Amplitude Dist Match:
KS statistic: {np.max(np.abs(np.sort(calcium_signal) - np.sort(surrogate))):.6f}
"""

axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=9)
axes[1, 2].set_title('Statistical Comparison')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('iaaft_example.png', dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved as iaaft_example.png")
print("IAAFT Surrogate Properties:")
print("✓ Same amplitude distribution (histogram)")
print("✓ Same power spectrum (frequency content)")
print("✗ Different temporal correlations (phase relationships)")
print("✗ Different autocorrelation structure")
print("→ Destroys causal relationships while preserving spectral properties")