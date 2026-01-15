import DOCTData
import numpy as np


def _apply_to_all(doct_obj, func):
    """Helper to apply a function to all matching arrays in results and meta."""
    for container in [doct_obj.results, doct_obj.meta]: # TODO: Edit this if more attributes are added later
        for key, value in container.items():
            if isinstance(value, np.ndarray):
                # 3D arrays (time series)
                if value.ndim == 3 and value.shape[1:] == doct_obj.data.shape[1:]:
                    container[key] = func(doct_obj.data, value)
                # 2D arrays (single frame results)
                elif value.ndim == 2 and value.shape == doct_obj.data.shape[1:]:
                    container[key] = func(doct_obj.data, value)


def binary_mask(DOCTData, threshold: int | float, invert = False, all = False, inplace = True) -> DOCTData:
    """ Apply binary thresholding to DOCTData.

    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object
    threshold: int | float
        Pixels with an average intensity below this number will be set to 0
    invert: bool, default False
        If True, pixels above the threshold are set to 0 instead of below.
    all : bool, default False
        If True, apply thresholding to all arrays in DOCTData with matching shape.
    inplace : bool, default True
        If True, modify DOCTData directly. If False, return a new thresholded copy.
    Returns
    -------
    DOCTData or None
        If inplace=False, returns new thresholded DOCTData.
        If inplace=True, modifies input and returns DOCTData.
    """
    if not inplace:
        DOCTData = DOCTData.copy(deep=True)
    
    if invert:
        DOCTData.data = np.where(DOCTData.data.mean(axis=0) > threshold, 0, DOCTData.data)
    else:
        DOCTData.data = np.where(DOCTData.data.mean(axis=0) < threshold, 0, DOCTData.data)
    
    if all == True:
        # Apply mask to all results arrays
        _apply_to_all(DOCTData, lambda data, result: 
            np.where(data[0] != 0, result, 0) if result.ndim == 2 
            else np.where(data != 0, result, 0)
        )

    if inplace: # Save metadata regarding modifications
        DOCTData.meta['binary_mask'] = f"Applied binary mask with threshold {threshold}, invert={invert}"
        DOCTData.meta[f"binary_mask_{threshold}_invert_{invert}"] = np.where(DOCTData.data[0] == 0 , 0, 1)

    return DOCTData

def brightness_gradient_correction(DOCTData, pct = 0.5, inplace = False) -> DOCTData:
    """Apply brightness gradient correction to DOCTData.
    The brightness gradient correction compensates for the typical decrease in brightness from top to bottom
    This is done by creating a gradient vector that goes from the specified percentile value from histogram
    to the minimum brightest value in the image. Then, each column of the image is subtracted by this gradient.

    Note: The author of this function wrote this for binary masking. From trial and error, they found 50% was a 
    good value to correct for this, determined by how well the binary mask successfully masked background pixels.
    
    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object
    pct : float, default 50%
        Percentage of histogram that you want 
    inplace : bool, default False
        If True, modify DOCTData directly. If False, return a new corrected copy.

    Returns
    -------
    DOCTData or None
        If inplace=False, returns new corrected DOCTData.
        If inplace=True, modifies input and returns DOCTData.
    """
    # Work on a copy if not modifying in place
    if not inplace:
        DOCTData = DOCTData.copy(deep=True)

    

    grad_vec = np.linspace(np.percentile(DOCTData.data, pct), np.min(DOCTData.data), DOCTData.data.shape[1]).reshape(-1,1)
    grad_mat = np.repeat(grad_vec, DOCTData.data.shape[2], axis = 1)
    DOCTData.data = np.clip(DOCTData.data - grad_mat, a_min=0, a_max=None)
    DOCTData.data = (255 * (DOCTData.data-np.min(DOCTData.data.mean(axis=0)))  / (np.max(DOCTData.data.mean(axis=0)) - np.min(DOCTData.data.mean(axis=0)))).astype(np.uint8)

    if inplace: # Save metadata regarding modifications
        DOCTData.meta['brightness_gradient_correction'] = "Applied brightness gradient correction."

    return DOCTData

def crop(DOCTData, top_left: tuple, bottom_right: tuple, all = False, inplace = True) -> DOCTData:
    """Crop DOCTData to region of interest.
    
    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object
    top_left : tuple
        (x, y) coordinates of top-left corner of crop
    bottom_right : tuple
        (x, y) coordinates of bottom-right corner of crop
    all : bool, default False
        If True, crop all arrays in DOCTData with matching shape.
    inplace : bool, default True
        If True, modify DOCTData directly. If False, return a new cropped copy.

    Returns
    -------
    DOCTData or None
        If inplace=False, returns new cropped DOCTData.
        If inplace=True, modifies input and returns DOCTData.

    Note: Stores the cropped coordinates in the meta dictionary.
    """
    # Work on a copy if not modifying in place
    if not inplace:
        DOCTData = DOCTData.copy(deep=True)
    
    if all == True:
        # Apply crop to all results arrays
        _apply_to_all(DOCTData, lambda data, result:
            result[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] if result.ndim == 2
            else result[:, top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        )
    else:
        DOCTData.data = DOCTData.data[:, top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    DOCTData.meta['cropped_coordinates'] = f"Top-left coordinates: {top_left}. Bottom-right: {bottom_right}"

    return DOCTData

def linear(DOCTData, inplace = False) -> DOCTData:
    """Apply linear scaling to DOCTData.
    
    Converts from dB scale to linear scale using: 10^(dB/20)
    
    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object (should be in dB scale, e.g., 0-80 dB)
    inplace : bool, default False
        If True, modify DOCTData directly. If False, return a new scaled copy.

    Returns
    -------
    DOCTData or None
        If inplace=False, returns new scaled DOCTData.
        If inplace=True, modifies input and returns DOCTData.
    """
    if not inplace:
        DOCTData = DOCTData.copy(deep=True)
    
    # Convert from dB to linear: 10^(dB/20)
    DOCTData.data = 10 ** (DOCTData.data.astype(np.float32) / 20)
    

    if inplace: # Save metadata regarding modifications
        DOCTData.meta['linear_scale'] = "True"
    if 'logarithmic_scale' in DOCTData.meta:
        del DOCTData.meta['logarithmic_scale']

    return DOCTData

def logarithm(DOCTData, dB = False, inplace = True) -> DOCTData:
    """Apply logarithmic scaling to DOCTData.
    
    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object
    inplace : bool, default True
        If True, modify DOCTData directly. If False, return a new scaled copy.

    Returns
    -------
    DOCTData or None
        If inplace=False, returns new scaled DOCTData.
        If inplace=True, modifies input and returns DOCTData.
    """
    if not inplace:
        DOCTData = DOCTData.copy(deep=True)
    
    if dB == False:
        DOCTData.data = 10 * np.log10(np.maximum(DOCTData.data.astype(np.float32), 1e-6)) #1e-6 to avoid log(0)
    else:
        DOCTData.data = 20 * np.log10(np.maximum(DOCTData.data.astype(np.float32), 1e-6)) #1e-6 to avoid log(0)

    if inplace: # Save metadata regarding modifications
        DOCTData.meta['logarithmic_scale'] = "Applied logarithmic scaling (10*log10)."
    if 'linear_scale' in DOCTData.meta:
        del DOCTData.meta['linear_scale']

    return DOCTData

def trim(DOCTData, start_frame: int, end_frame: int, all = False, inplace = True) -> DOCTData:
    """Trim DOCTData to specified frame range.
    
    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object
    start_frame : int
        Starting frame index (inclusive)
    end_frame : int
        Ending frame index (exclusive)
    all : bool, default False
        If True, trim all arrays in DOCTData with matching shape.
    inplace : bool, default True
        If True, modify DOCTData directly. If False, return a new trimmed copy.

    Returns
    -------
    DOCTData or None
        If inplace=False, returns new trimmed DOCTData.
        If inplace=True, modifies input and returns DOCTData.

    Note: Stores the trimmed frame range in the meta dictionary.
    """
    if not inplace:
        DOCTData = DOCTData.copy(deep=True)
    
    if all == True:
        _apply_to_all(DOCTData, lambda data, result:
            result[start_frame:end_frame] if result.ndim == 3
            else result
        )
    else:
        DOCTData.data = DOCTData.data[start_frame:end_frame]

    DOCTData.meta['trimmed_frames'] = f"Trimmed to frames {start_frame} to {end_frame}."

    return DOCTData

