import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from preprocessing import *

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

