import numpy as np
from numpy import fft as sp_fft
import dask.array as da


def make_chunks(x: np.ndarray, nperseg: int, noverlap: int):
    """rechuncks data efficiently using strides"""
    step = nperseg - noverlap
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result  # result is N x K x T

def _triage_segments(window, nperseg: int, input_length: int):
    from scipy.signal.windows import get_window
    import warnings

    """
    Parses window and nperseg arguments for spectrogram and _spectral_helper.
    This is a helper function, not meant to be called externally.
    Parameters
    ----------
    window : string, tuple, or ndarray # default is "hann"
        If window is specified by a string or tuple and nperseg is not
        specified, nperseg is set to the default of 256 and returns a window of
        that length.
        If instead the window is array_like and nperseg is not specified, then
        nperseg is set to the length of the window. A ValueError is raised if
        the user supplies both an array_like window and a value for nperseg but
        nperseg does not equal the length of the window.
    nperseg : int
        Length of each segment
    input_length: int
        Length of input signal, i.e. x.shape[-1]. Used to test for errors.
    Returns
    -------
    win : ndarray
        window. If function was called with string or tuple than this will hold
        the actual array used as a window.
    nperseg : int
        Length of each segment. If window is str or tuple, nperseg is set to
        256. If window is array_like, nperseg is set to the length of the
        window.
    """
    # parse window; if array like, then set nperseg = win.shape
    if isinstance(window, str) or isinstance(window, tuple):
        # if nperseg not specified
        if nperseg is None:
            nperseg = 256  # then change to default
        if nperseg > input_length:
            warnings.warn(
                "nperseg = {0:d} is greater than input length "
                " = {1:d}, using nperseg = {1:d}".format(nperseg, input_length)
            )
            nperseg = input_length
        win = get_window(window, nperseg)  # return a 1d array
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError("window must be 1-D")
        if input_length < win.shape[-1]:
            raise ValueError("window is longer than input signal")
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError(
                    "value specified for nperseg is different" " from length of window"
                )
    return win, nperseg


def estimate_spectrum(
    xnt: np.array,
    ynt=None,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="spectrum",  # default scaling from scipy
    axis=-1,
    freq_minmax=[0, np.inf],
    abs=False,
    alltoall=True,  # every x with every y
    return_coefs=False,
    y_in_coefs=False,
    x_in_coefs=False,
):
    """Calculate various forms of windowed FFTs for PSD, CSD, etc.

    This is a helper function that implements the commonality between
    the stft, psd, csd, and spectrogram functions. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Parameters
    ----------
    x : array_like
        Array or sequence containing the data to be analyzed.
    y : array_like
        Array or sequence containing the data to be analyzed. If this is
        the same object in memory as `x` (i.e. ``_spectral_helper(x,
        x, ...)``), the extra computations are spared.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross
        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`
        and `y` are measured in V and `fs` is measured in Hz.
        Defaults to 'density'
    axis : int, optional
        Axis along which the FFTs are computed; the default is over the
        last axis (i.e. ``axis=-1``).

    Returns
    -------
    freqs : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of times corresponding to each data segment
    result : ndarray
        Array of output data, contents dependent on *mode* kwarg.

    """

    if not x_in_coefs:
        if len(xnt.shape) < 2:
            xnt = xnt[None]
        xn = xnt.shape[0]
        coefs_xnkf, freqs, win = compute_spectral_coefs(
            xnt=xnt,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            axis=axis,
            freq_minmax=freq_minmax,
            nfft=nfft,
        )

    else:
        xn = xnt.shape[0]
        coefs_xnkf = xnt  # assume xnt is already N x K x F
        f = coefs_xnkf.shape[-1]
        freqs = sp_fft.rfftfreq(f * 2 - 1, d=1 / fs)  # to check

    if ynt is not None:
        if not y_in_coefs:
            if len(ynt.shape) < 2:
                ynt = ynt[None]
            yn = ynt.shape[0]
            coefs_ynkf, _, _ = compute_spectral_coefs(
                xnt=ynt,
                fs=fs,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend=detrend,
                return_onesided=return_onesided,
                scaling=scaling,
                axis=axis,
                freq_minmax=freq_minmax,
                nfft=nfft,
            )

        else:
            yn = ynt.shape[0]
            coefs_ynkf = ynt
        if not alltoall:
            if yn != xn:
                raise Exception("invalid dimensions for not all to all")

    wn = coefs_xnkf.shape[-1]
    kn = coefs_xnkf.shape[1]
    coefs_xnkf = da.from_array(coefs_xnkf, chunks=("auto"))
    if ynt is not None:
        coefs_ynkf = da.from_array(coefs_ynkf, chunks=("auto"))
        if alltoall:
            # pxy = jnp.einsum("nkf, mkf-> nmf", coefs_xnkf, np.conj(coefs_ynkf))
            pxy = da.einsum("nkf, mkf-> nmf", coefs_xnkf, np.conj(coefs_ynkf), optimize = True)
        else:
            # pxy = jnp.einsum("nkf, nkf-> nf", coefs_xnkf, np.conj(coefs_ynkf))
            pxy = da.einsum("nkf, nkf-> nf", coefs_xnkf, np.conj(coefs_ynkf), optimize = True)

    else:
        if alltoall:
            # pxy = jnp.einsum("nkf, mkf-> nmf", coefs_xnkf, np.conj(coefs_xnkf))
            pxy = da.einsum("nkf, mkf-> nmf", coefs_xnkf, da.conj(coefs_xnkf), optimize = True)
        else:
            # pxy = jnp.einsum("nkf, nkf-> nf", coefs_xnkf, np.conj(coefs_xnkf))
            pxy = da.einsum("nkf, nkf-> nf", coefs_xnkf, da.conj(coefs_xnkf), optimize = True)

    pxy /= (kn*(win**2).sum())
 
    if abs:
        pxy = np.abs(pxy).real
    if scaling == "density":
        pxy /= fs
    

        
    if return_coefs:
        if ynt is not None:
            return pxy, freqs, coefs_xnkf, coefs_ynkf
        else:
            return pxy, freqs, coefs_xnkf
    else:
        return pxy, freqs


def get_freqs(nperseg: int, fs: float, freq_minmax=[0, np.inf]):
    freqs = sp_fft.rfftfreq(nperseg, d=1 / fs)  # to check
    valid_freq_inds = (np.abs(freqs) >= freq_minmax[0]) & (
        np.abs(freqs) <= freq_minmax[1]
    )
    valid_freqs = freqs[valid_freq_inds]
    return freqs, valid_freqs, valid_freq_inds


def compute_multiscale_spectral_coefs(xnt: np.ndarray, fs: float, window: str, noverlap: int, detrend: str, return_onesided: bool, scaling: str, axis: int, num_levels: int = 10, reps: int = 3):
    """Compute Spectral Fourier Coefs for a multiscale analysis"""

    def detrend_func(d):
        from scipy.signal import signaltools
        return signaltools.detrend(d, type=detrend, axis=-1)

    n, t = xnt.shape
    N = t
    freqs = get_freqs(t, fs)
    lowest_freq = 1/(N/fs)
    highest_freq = (N/2)/(N/fs) # i.e.: 2/fps
    freqs_ = (np.arange(1, int(N/2))/(N/fs))[::(int(N/2)//num_levels)]



    spectra = []
    freqs_all = []
    for ind, freq in enumerate(freqs_):
        frame_by_cycle = (1/freq)*fs
        nperseg = int(frame_by_cycle)*reps
        # print(f"freq: {freq}, nperseg: {nperseg}")
        win, nperseg = _triage_segments(window, nperseg, input_length=t) 
        nfft = nperseg # win is a 1d array
        noverlap = nperseg // 2

        if noverlap >= nperseg:
            print(f"noverlap: {noverlap}, nperseg: {nperseg}")
            raise ValueError("noverlap must be less than nperseg.")

        if return_onesided:
            if np.iscomplexobj(xnt):
                sides = "twosided"
            else:
                sides = "onesided"
        else:
            sides = "twosided"

        if sides == "twosided":
            freqs = sp_fft.fftfreq(nfft, 1 / fs)
        elif sides == "onesided":
            freqs = sp_fft.rfftfreq(nfft, 1 / fs)


        if detrend is False:
            detrend_func = None
        coefs_xnkf = myfft_helper(xnt, win, detrend_func, nperseg, noverlap, nfft, sides)
        spectra.append(coefs_xnkf)
        freqs_all.append(freqs)

    return spectra, freqs_all


  
def wrangle_multiscale_coefs(coefs_xnkfs, freqs_):
    from scipy.interpolate import interp1d
    min_freq = 100
    final_freqs = []
    final_coefs_xnkfs = []
    for ind, freqs in enumerate(freqs_[::-1]):
        freqs = freqs[1:]
        vinds = np.argwhere(freqs<min_freq)[:,0]
        if len(vinds) > 0:
            freq_v = freqs[vinds]
            min_freq = np.min(freq_v)
            coefs_xnkf = coefs_xnkfs[::-1][ind][0, :, 1:][:, vinds]
            final_freqs.append(freq_v)
            final_coefs_xnkfs.append(np.abs(coefs_xnkf))
    final_freqs = np.hstack(final_freqs[::-1])
    final_coefs_xnkfs = final_coefs_xnkfs[::-1]
    
    max_len = final_coefs_xnkfs[-1].shape[0]
    coefs_interp = []
    for ind, coefs_xnkf in enumerate(final_coefs_xnkfs):
        coefs_xnkf_interp = np.zeros((max_len, coefs_xnkf.shape[1]))
        # interpolate the coefs_xnkf to the max_len using closest neighbor
        for i in range(coefs_xnkf.shape[1]):
            # Use scipy.interpolate.interp1d for nearest neighbor interpolation
            f_interp = interp1d(np.linspace(0, 1, coefs_xnkf.shape[0]), coefs_xnkf[:, i], kind="nearest", bounds_error=False, fill_value="extrapolate")
            coefs_xnkf_interp[:, i] = f_interp(np.linspace(0, 1, max_len))
            
        coefs_interp.append(coefs_xnkf_interp)
    coefs_interp = np.hstack(coefs_interp)


    min_freq = final_freqs[0]
    max_freq = final_freqs[-1]
    spectrogram = coefs_interp.T[::-1]
    return spectrogram, final_freqs, min_freq, max_freq  

def compute_spectral_coefs(  # used in coherence
    xnt: np.ndarray,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    detrend="constant",
    return_onesided=True,
    scaling="spectrum",
    axis=-1,
    freq_minmax=[0, np.inf],
    nfft=None,
):
    """Compute Spectral Fourier Coefs"""

    def detrend_func(d):
        from scipy.signal import signaltools

        return signaltools.detrend(d, type=detrend, axis=-1)

    n, t = xnt.shape
    win, nperseg = _triage_segments(
        window, nperseg, input_length=t
    )  # win is a 1d array

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError("nfft must be greater than or equal to nperseg.")
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg.")

    if return_onesided:
        if np.iscomplexobj(xnt):
            sides = "twosided"
        else:
            sides = "onesided"
    else:
        sides = "twosided"

    if sides == "twosided":
        freqs = sp_fft.fftfreq(nfft, 1 / fs)
    elif sides == "onesided":
        freqs = sp_fft.rfftfreq(nfft, 1 / fs)


    if detrend is False:
        detrend_func = None
    coefs_xnkf = myfft_helper(xnt, win, detrend_func, nperseg, noverlap, nfft, sides)

    valid_freqs = (np.abs(freqs) >= freq_minmax[0]) & (np.abs(freqs) <= freq_minmax[1])
    freqs = freqs[valid_freqs]
    coefs_xnkf = coefs_xnkf[:, :, valid_freqs]

    return coefs_xnkf, freqs, win


def myfft_helper(
    x: np.ndarray,
    win: np.ndarray,
    detrend_func,
    nperseg: int,
    noverlap: int,
    nfft,
    sides: str,
):
    """
    Calculate windowed FFT, for internal use by
    `scipy.signal._spectral_helper`.
    This is a helper function that does the main FFT calculation for
    `_spectral helper`. All input validation is performed there, and the
    data axis is assumed to be the last axis of x. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.
    Returns
    -------
    result : ndarray
        Array of FFT data
    Notes
    -----
    Adapted from matplotlib.mlab
    .. versionadded:: 0.16.0
    """
    # Created strided array of data segments
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        result = make_chunks(x, nperseg, noverlap)

    # Detrend each data segment individually
    if detrend_func is not None:
        result = detrend_func(result)  # default to last axis - result is N x K x T

    # Apply window by multiplication
    result = win * result # vmsr

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    if sides == "twosided":
        func = sp_fft.fft
    else:
        result = (
            result.real
        )  # forces result to be real (should be real anyway as doing one sided fft...)
        func = sp_fft.rfft
    coefs_nkf = func(result, n=nfft)

    return coefs_nkf

