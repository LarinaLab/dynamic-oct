import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from preprocessing import *

def generate_RgbImage(doct, dbInt, doctRange, octRange):
    """
    Code from R. Morishita et.al., arXiv 2412.09351 (2024)

    Parameters
    ----------
    doct : 3D array
       aLIV or Swiftness image
    dbInt : 3D array
        dB-scale OCT intensity image
    doctRange : 1D tuple (min, max)
        dynamic range of aLIV, which is used as hue of pseudo-color image
    octRange : 1D tuple (min, max)
        dynamic range of dB-scaled OCT intensity, which is used as brightness of pseudo-color image

    Returns
    -------
    rgbImage : RGB array of pseudo-color image
        pseudo-color image

    """
    scale_clip = lambda data, vmin, vmax, scale=1.0: np.clip((data-vmin)*(scale/(vmax-vmin)), 0, scale)
    hsvImage = np.stack([scale_clip(doct, *doctRange, 0.33), np.ones_like(doct),
                       scale_clip(dbInt, *octRange)], axis=-1)
    rgbImage = hsv_to_rgb(hsvImage)
    return rgbImage

def show_rgb(result, title="result"):
    """Input: HxWx3 ndarray. Output, an RGB image"""
    

def show_heatmap(result, display_image=None, title="result", output_path=None, filename=None):
    """Show and optionally save heatmap with colorbar"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if display_image is not None:
        ax.imshow(display_image, cmap='gray')
        im = ax.imshow(result, cmap='viridis', interpolation='nearest')
    else:
        im = ax.imshow(result, cmap='viridis', interpolation='nearest')
    
    ax.set_title(f'{title}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.5)
    
    if output_path and filename:
        plt.savefig(os.path.join(output_path, filename), bbox_inches='tight', dpi=300)
        print(f"Saved {title} heatmap to {filename}")
    
    plt.show()
    return fig

