import DOCTData
import numpy as np
import scipy
import cv2
from visual import generate_RgbImage
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter
from preprocessing import trim
from joblib import Parallel, delayed
from rgb_by_ngc_cpu import rgb_by_neural_gas_b_is_dc_hist

def apply_windowed(DOCTData, method, window_size, step_size, inplace=True, n_jobs=1, **method_kwargs):
    """
    Apply any analysis method to sliding windows along the time axis.
    
    This function enables processing large time-series data in manageable chunks,
    useful for volumetric acquisitions as done in Umezu K et al. Biomed Opt Express 13 (2022).
    Windows are processed in parallel for faster computation.
    
    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object with shape (n_frames, height, width)
    method : callable
        Analysis function that takes DOCTData and returns DOCTData with results.
        Examples: std, liv, ocds
    window_size : int
        Number of frames per window
    step_size : int
        Stride between consecutive windows (use window_size for no overlap)
    inplace : bool, default True
        If True, stores results in input object. If False, returns new DOCTData.
    n_jobs : int, default -1
        Number of parallel jobs. -1 uses all CPU cores.
        Set to 1 for sequential processing (debugging).
    **method_kwargs : dict
        Additional keyword arguments to pass to the method
        
    Returns
    -------
    DOCTData
        Object with results stored as 3D arrays (n_windows, height, width) for each metric.
        Original data remains unchanged.
        
    Examples
    --------
    >>> # Apply STD to 10,000 frame dataset in 256-frame windows with 50% overlap
    >>> d_data = apply_windowed(d_data, std, window_size=256, step_size=128)
    >>> d_data.results["std_3d"].shape  # (78, height, width) for 10k frames
    
    >>> # Apply OCDS with custom tau parameter
    >>> d_data = apply_windowed(d_data, ocds, window_size=256, step_size=256, tau=(5,100))
    
    Notes
    -----
    - Results are stacked along axis 0, creating pseudo-3D volumes
    - For methods returning multiple results (like ocds with different taus),
      each result key will have shape (n_windows, height, width)
    - Result keys are automatically suffixed with '_3d' to distinguish from 2D results
    - Parallel processing significantly speeds up computation for large datasets
    """
    if not inplace:
        DOCTData = DOCTData.copy(deep=True)
    
    n_frames = DOCTData.data.shape[0]
    
    # Define function to process a single window
    def process_window(start):
        """Process a single window and return its results"""
        windowed_obj = trim(DOCTData, start_frame=start, end_frame=start + window_size, inplace=False)
        windowed_obj.results = {}
        windowed_result = method(windowed_obj, inplace=True, **method_kwargs)
        return start, windowed_result.results
    
    # Get all window start positions
    window_starts = list(range(0, n_frames - window_size + 1, step_size))
    
    # Process windows in parallel
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(process_window)(start) 
        for start in tqdm(window_starts, desc="Processing windows")
    )
    
    # Accumulate results in order
    accumulated_results = {}
    for start, window_results in sorted(results_list, key=lambda x: x[0]):
        for key, value in window_results.items():
            if key not in accumulated_results:
                accumulated_results[key] = []
            accumulated_results[key].append(value)
    
    # Stack all results into 3D arrays and save with _3d suffix
    for key, value_list in accumulated_results.items():
        DOCTData.results[f"{key}_3d"] = np.stack(value_list, axis=0)
    
    # Store metadata
    DOCTData.meta['windowed'] = True
    DOCTData.meta['window_size'] = window_size
    DOCTData.meta['step_size'] = step_size
    
    return DOCTData

def frequency_binning(DOCTData, frame_rate, range : tuple, window : str = None, normalize = True, inplace=True) -> DOCTData:
    """
    This function performs frequency binning for a single frequency range.
    
    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object
    frame_rate : float
        Frame rate in Hz
    range : tuple
        Single frequency bin (min, max) in units of Hz.
        Example: (0.1, 3.0)
    window : str or None
        Type of window to apply before FFT. Options: 'hann' or None.
        Default is None (no window).
    normalize : bool, default True
        If True, normalize the frequency bin to [0, 1].
    inplace : bool, default True
        If True, modify DOCTData directly. If False, return a new DOCTData with frequency binned data.
        
    Returns
    -------
    DOCTData
        If inplace=False, returns new DOCTData with frequency_binning data.
        If inplace=True, modifies input and returns DOCTData.      
    """
    if not inplace:
        DOCTData = DOCTData.copy(deep=True)

    I = DOCTData.data - np.mean(DOCTData.data, axis=0)  # Remove DC component

    if window == 'hann':
        hann_window = np.hanning(I.shape[0])[:, np.newaxis, np.newaxis]
        I = I * hann_window
        # Zero padding
        pad_length = 2**int(np.ceil(np.log2(I.shape[0]))) - I.shape[0]
        I = np.pad(I, ((0, pad_length), (0, 0), (0, 0)), mode='constant')

    fft_result = np.fft.fft(I, axis=0)
    fft_freqs = np.fft.fftfreq(I.shape[0], d=1/frame_rate)
    fft_magnitude = 2*np.abs(fft_result)  # Note: taking all frequencies

    # Extract single frequency bin
    min_freq, max_freq = range
    channel_data = np.mean(fft_magnitude[(fft_freqs >= min_freq) & (fft_freqs <= max_freq)], axis=0)
    
    if normalize:
        # Enhancing contrast by clipping extreme values, as done in R. Morishita et.al., arXiv 2412.09351 (2024)
        unique_vals = np.unique(channel_data)
        top_limit = np.max(unique_vals[:int(unique_vals.size * 0.9999)])
        bottom_limit = np.min(unique_vals[int(unique_vals.size * 0.001):])
        channel_data = np.clip(channel_data, bottom_limit, top_limit)
    
        # Rescale to [0, 1]
        old_min = np.min(channel_data)
        old_max = np.max(channel_data)
        channel_data = (channel_data - old_min) / (old_max - old_min)

    # Save single channel
    DOCTData.results[f"frequency_binning={range}"] = channel_data

    return DOCTData

def RGB_frequency_binning(DOCTData, frame_rate, RGB : list = [ (0.1, 0.3), (0.3, 0.6), (0.6, 0.9) ], window : str = None, normalize = True,inplace=True) -> DOCTData:
    """
    This function performs frequency binning and creates an RGB matrix. 
    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object
    frame_rate : float
    RGB : list of tuples
        Frequency bins for R, G, B channels in units of Hz.
        Example: [(0.1, 0.3), (0.3, 0.6), (0.6, 0.9)]
    window : str or None
        Type of window to apply before FFT. Options: 'hann' or None. (These are current ones used in literature. Please feel free to experiment with others)
        Default is None (no window).
    normalize : bool, default True
        If True, normalize each frequency bin to [0, 1].
    inplace : bool, default True
        If True, modify DOCTData directly. If False, return a new DOCTData with frequency binned data.
    Returns
    -------
    DOCTData or None
        If inplace=False, returns new DOCTData with frequency_binning ((R, G, B) x H x W) data.
        If inplace=True, modifies input and returns DOCTData.      
    """
    if not inplace:
        DOCTData = DOCTData.copy(deep=True)

    I = DOCTData.data - np.mean(DOCTData.data, axis=0)  # Remove DC component

    if window == 'hann':
        hann_window = np.hanning(I.shape[0])[:, np.newaxis, np.newaxis]
        I = I * hann_window
        # Zero padding
        pad_length = 2**int(np.ceil(np.log2(I.shape[0]))) - I.shape[0]
        I = np.pad(I, ((0, pad_length), (0, 0), (0, 0)), mode='constant')


    fft_result = np.fft.fft(I, axis=0)
    fft_freqs = np.fft.fftfreq(I.shape[0], d=1/frame_rate)
    fft_magnitude = 2*np.abs(fft_result) # Note: taking all frequencies
    
    # Use only positive frequencies (match torch implementation)
    pos_side = I.shape[0] // 2 + 1
    fft_magnitude = fft_magnitude[:pos_side]
    fft_freqs = np.abs(fft_freqs[:pos_side])  # abs() to handle -0.0

    rgb_image = np.zeros((3, I.shape[1], I.shape[2]), dtype=np.float32)
    for channel, (min, max) in enumerate(RGB):
        # Select frequencies in range
        freq_mask = (fft_freqs >= min) & (fft_freqs <= max)
        
        # Check if any frequencies match (handles empty DC-only blue channel)
        if np.any(freq_mask):
            rgb_image[channel, :, :] = np.mean(fft_magnitude[freq_mask], axis=0)
        else:
            # No frequencies in this range, leave as zeros
            rgb_image[channel, :, :] = 0.0
        
        if normalize:
            # Check if channel has any non-zero variation before normalizing
            unique_vals = np.unique(rgb_image[channel, :, :])
            if unique_vals.size > 1:  # More than just one value (not all zeros/constant)
                # Enhancing contrast by clipping extreme values, as done in R. Morishita et.al., arXiv 2412.09351 (2024)
                top_idx = int(unique_vals.size * 0.99)
                bottom_idx = int(unique_vals.size * 0.01)
                
                if top_idx < unique_vals.size and bottom_idx < unique_vals.size:
                    top_limit = np.max(unique_vals[:top_idx]) if top_idx > 0 else unique_vals[0]
                    bottom_limit = np.min(unique_vals[bottom_idx:])
                    rgb_image[channel, :, :] = np.clip(rgb_image[channel, :, :], bottom_limit, top_limit)
            
                    # Rescale to [0, 1]
                    old_min = np.min(rgb_image[channel, :, :])
                    old_max = np.max(rgb_image[channel, :, :])
                    if old_max > old_min:  # Avoid divide by zero
                        rgb_image[channel, :, :] = (rgb_image[channel, :, :] - old_min) / (old_max - old_min)

    # Save each channel separately
    DOCTData.results[f"frequency_binning_R={RGB[0]}"] = rgb_image[0]
    DOCTData.results[f"frequency_binning_G={RGB[1]}"] = rgb_image[1]
    DOCTData.results[f"frequency_binning_B={RGB[2]}"] = rgb_image[2]

    return DOCTData
    
def liv(DOCTData, inplace=True) -> DOCTData:
    """
    Calculate the logarithmic intensity variance (LIV) per pixel of the time series.
    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object
    inplace : bool, default True
        If True, modify DOCTData directly. If False, return a new DOCTData with LIV data.
    Returns
    -------
    DOCTData or None
        If inplace=False, returns new DOCTData with LIV data.
        If inplace=True, modifies input and returns DOCTData.
    """
    if not inplace:
        DOCTData = DOCTData.copy(deep=True)

    DOCTData.results["liv"] = np.var(DOCTData.data, axis=0)

    return DOCTData

def ocds(DOCTData, frame_rate: float, tau: tuple = (1, 50), inplace=True):
    """
    Calculate the OCT (auto)correlation decay speed per pixel of the time series.
    
    This version includes mean subtraction and normalization for proper autocorrelation analysis.
    
    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object
    tau : tuple, default (1, 50)
        Time lag range (in frames) for computing the decay slope
    inplace : bool, default True
        If True, modify DOCTData directly. If False, return a new DOCTData with OCDS data.
    Returns
    -------
    DOCTData or None
        If inplace=False, returns new DOCTData with OCDS data.
        If inplace=True, modifies input and returns DOCTData.
    """
    if not inplace:
        DOCTData = DOCTData.copy(deep=True)
    
    I = DOCTData.data.reshape(DOCTData.n_frames, -1) # (T, H*W)
    
    
    # Subtract mean from each pixel's time series to remove DC bias
    I = I - np.mean(I, axis=0, keepdims=True)
    
    rho_A = []
    for pixel in range(0, I.shape[1]):
        # Compute autocorrelation for each pixel
        rho_A_pixel = scipy.signal.correlate(I[:, pixel], I[:, pixel], mode='full', method='fft')
        # Normalize by zero-lag autocorrelation (center value) so it starts at 1.0
        zero_lag = rho_A_pixel[len(rho_A_pixel)//2]
        if zero_lag != 0:
            rho_A_pixel /= zero_lag
        else:
            rho_A_pixel = np.zeros_like(rho_A_pixel)
        rho_A.append(rho_A_pixel)
    
    rho_A = np.stack(rho_A, axis=1) # (2T-1, H*W)
    
    # Extract positive lags in tau range and fit slope
    time_lags = np.arange(tau[0], tau[1]) / DOCTData.meta['frame_rate']
    autocorr_slice = rho_A[rho_A.shape[0]//2+tau[0]:rho_A.shape[0]//2+tau[1], :] # Taking corresponding positive lag values only
    
    # polyfit returns [slope, intercept], we want slope (first coefficient)
    ocds = np.polyfit(time_lags, autocorr_slice, deg=1)[0]
    ocds = ocds.reshape(DOCTData.data.shape[1], DOCTData.data.shape[2])
    
    DOCTData.results[f"ocds{tau}"] = ocds
    
    return DOCTData

def std(DOCTData, inplace=True) -> DOCTData:
    """
    Calculate the standard deviation per pixel of the time series.
    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object
    inplace : bool, default True
        If True, modify DOCTData directly. If False, return a new DOCTData with std data.
    Returns
    -------
    DOCTData or None
        If inplace=False, returns new DOCTData with std data.
        If inplace=True, modifies input and returns DOCTData.
    """
    if not inplace:
        DOCTData = DOCTData.copy(deep=True)

    DOCTData.results["std"] = np.std(DOCTData.data, axis=0)

    return DOCTData