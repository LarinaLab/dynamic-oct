def alpha_r2(DOCTData, inplace=True, n_jobs=1) -> DOCTData:
    """
    Calculate alpha and R² metrics per pixel based on power spectrum fitting.
    
    This implements Monfort's method which fits a power-law model (a*f^(-b) + c) 
    to the power spectrum of each pixel's time series.
    
    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object with linear intensity data
    frame_rate : float
        Frames per second (used for frequency scaling)
    smooth_box_size : int, default 5
        Size of 3D box filter for smoothing power spectra (odd number recommended)
    inplace : bool, default True
        If True, modify DOCTData directly. If False, return a new DOCTData.
    n_jobs : int, default -1
        Number of parallel jobs for fitting. -1 uses all CPU cores.
        Set to 1 for sequential processing (debugging).
        
    Returns
    -------
    DOCTData
        Object with results["alpha"] and results["r2"] added.
        - alpha: Power-law exponent (higher = more low-frequency dominated)
        - r2: Goodness of fit (0-1, higher = better fit quality)
        
    Notes
    -----
    - Input data should be in linear scale, not dB
    - Power fluctuation correction is applied: I(t) = I(t)/mean(I) - 1
    - Only positive frequencies [2, Nyquist] are used for fitting
    - Weighted least squares fitting uses power spectrum values as weights
    
    References
    ----------
    Tual Monfort
    https://github.com/noahheldt/A-guide-to-dynamic-OCT-data-analysis
    """
    if not inplace:
        DOCTData = DOCTData.copy(deep=True)

    I = DOCTData.data / np.mean(DOCTData.data, axis=0) - 1  # Power fluctuation correction

    fft = 2 * np.abs(np.fft.fft(I, axis=0))
    fft = fft[1:(fft.shape[0]//2), :, :]  # (freq, H, W) - keep 3D
    f = np.fft.fftfreq(I.shape[0], d=1/DOCTData.meta['frame_rate'])[1:(I.shape[0]//2)]
    
    # Smooth power spectra with 3x3 spatial box filter (paper notes it is averaging over 8 neighboring pixels).
    # The matlab code in the repo uses a 3D box filter, which is not represented in the paper.
    # I chose to implement the paper's idea.
    fft = uniform_filter(fft, size=(1, 3, 3), mode='nearest')
    # Reshape to (freq, H*W) for pixel-wise fitting
    fft = fft.reshape(fft.shape[0], -1)

    def power_law(f, a, b, c):
        return a * (f ** (-b)) + c
    
    def fit_pixel(pixel):
        """Fit power law to a single pixel - for parallel processing"""
        P = fft[:, pixel]
        
        # Skip pixels with NaN, Inf, or all zeros
        if not np.all(np.isfinite(P)) or np.all(P == 0):
            return pixel, 0.0, 0.0
        
        try:
            popt, pcov = curve_fit(power_law, f, P, p0=[1, 1, 1], bounds = [0, np.inf])
            alpha = popt[0]
            
            # Calculate R²
            residuals = P - power_law(f, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((P - np.mean(P))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return pixel, alpha, r2
        except (RuntimeError, ValueError):
            # Fit failed, return zeros
            return pixel, 0.0, 0.0
    
    # Initialize flattened result arrays
    n_pixels = fft.shape[1]  # Number of pixels (H*W)
    alpha_flat = np.zeros(n_pixels, dtype=np.float32)
    r2_flat = np.zeros(n_pixels, dtype=np.float32)
    
    # Parallel fitting with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_pixel)(pixel) for pixel in tqdm(range(n_pixels), desc="Fitting power law per pixel", unit='pixels')
    )
    
    # Unpack results into arrays
    for pixel, alpha, r2 in results:
        alpha_flat[pixel] = alpha
        r2_flat[pixel] = r2
    
    # Reshape results back to 2D
    DOCTData.results['alpha'] = alpha_flat.reshape(DOCTData.data.shape[1], DOCTData.data.shape[2])
    DOCTData.results['r2'] = r2_flat.reshape(DOCTData.data.shape[1], DOCTData.data.shape[2])

    return DOCTData

def motility(DOCTData, frame_rate: float, inplace=True) -> DOCTData:
    """
    Calculate Monfort's motility metric per pixel.
    
    This metric quantifies temporal fluctuations
    
    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object
    frame_rate : float
        Frames per second (for metadata)
    inplace : bool, default True
        If True, modify DOCTData directly.
        
    Returns
    -------
    DOCTData
        
    References
    ----------
    Tual Monfort
    https://github.com/noahheldt/A-guide-to-dynamic-OCT-data-analysis/blob/main/motility_based_metrics/doct_monfort_motility
    """
    if not inplace:
        DOCTData = DOCTData.copy(deep=True)
    
    I = DOCTData.data  # (T, H, W)
    
    # Mean product of consecutive frames
    Ga = np.mean(I[:-1] * I[1:], axis=0)
    
    # Mean intensity
    mean_I = np.mean(I, axis=0)
    
    # Monfort's motility metric
    motility = np.abs(np.sqrt(Ga - mean_I**2) / mean_I)
    
    DOCTData.results["motility"] = motility
    
    return DOCTData