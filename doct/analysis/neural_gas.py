from ..DOCTData import DOCTData
import numpy as np
from tqdm import tqdm
import pathlib
import plotly.graph_objects as go
import tifffile
# import torch  # CHANGED: Removed torch import
from plotly.subplots import make_subplots
from skimage.exposure import match_histograms
# import matplotlib.pyplot as plt  # CHANGED: Added for save_uint8 function

"""
All code in this file is to perform RGB frequency binning using neural gas clustering.
This code is from the following Github repository:
https://github.com/noahheldt/A-guide-to-dynamic-OCT-data-analysis/tree/main
which has an associated publication:
Heldt N, Monfort T, Morishita R, Schönherr R, Thouvenin O, El-Sadek IA, König P, Hüttmann G, Grieve K, Yasuno Y. Guide to dynamic OCT data analysis. Biomed Opt Express. 2025 Oct 31;16(11):4851-4870. doi: 10.1364/BOE.571394. PMID: 41293693; PMCID: PMC12642996.

Code was converted to numpy operators by Beller J for accessibility to machines without GPU and to be compatible with software written with numpy > 2.0.
"""

def rgb_by_neural_gas_b_is_dc_hist(
    fft: np.ndarray,  # CHANGED: torch.Tensor -> np.ndarray
    freqs: np.ndarray,  # CHANGED: torch.Tensor -> np.ndarray
    n_epochs: int,
    init_lr: float = 1.0,
    end_lr: float = 1e-3,
    rel_inti_rad: float = 0.5,
    end_rad: float = 0.01,
    make_plots: bool = False,
    display_plots: bool = False,
) -> tuple[np.ndarray, dict[str, tuple[float, float]] | go.Figure]:  # CHANGED: torch.Tensor -> np.ndarray
    """Neural gas clustering to perform RGB frequency binning.

    Args:
        fft (np.ndarray): Positive side of the FFT.  # CHANGED: torch.Tensor -> np.ndarray
        freqs (np.ndarray): Corresponding frequencies.  # CHANGED: torch.Tensor -> np.ndarray
        n_epochs (int): Number of epochs to run.
        init_lr (float, optional): Initial learning rate. Defaults to 1.0.
        end_lr (float, optional): End learning rate to converge to. Defaults to 1e-3.
        rel_inti_rad (float, optional): Initial neighborhood radius. Defaults to 0.5.
        end_rad (float, optional): End neighborhood radius to converge to. Defaults to 0.01.
        make_plots (bool, optional): If spectral clustering plots should be saved. Defaults to False.
        display_plots (bool, optional): If spectral clustering plots should also be shown. Defaults to False.

    Returns:
        tuple[np.ndarray, dict[str, tuple[float, float]] | go.Figure]: RGB frequency binned image and either optimal bins dict or spectral plots figure.  # CHANGED: torch.Tensor -> np.ndarray
    """
    # Sum spectra
    fft_sum: np.ndarray = np.sum(fft, axis=-1)  # CHANGED: torch.Tensor -> np.ndarray, dim -> axis, torch.sum -> np.sum
    while fft_sum.ndim > 1:
        fft_sum = np.sum(fft_sum, axis=-1)  # CHANGED: torch.sum -> np.sum, dim -> axis

    # Remove 0 freq
    fft_sum_no_dc: np.ndarray = fft_sum[1:].copy().astype(fft_sum.dtype)  # CHANGED: torch.Tensor -> np.ndarray, torch.clone -> .copy(), .type_as -> .astype
    fft_sum_no_dc = rescale_range(fft_sum_no_dc, new_min=1e-3, new_max=1.0)
    freqs_no_dc: np.ndarray = freqs[1:].copy().astype(fft_sum.dtype)  # CHANGED: torch.Tensor -> np.ndarray, torch.clone -> .copy(), .type_as -> .astype

    # Create Nx2 matrix with frequency bins and corresponding amplitudes
    samples: np.ndarray = np.stack([freqs_no_dc, fft_sum_no_dc], axis=0).T  # CHANGED: torch.Tensor -> np.ndarray, torch.transpose(torch.stack(...), 0, 1) -> np.stack(...).T

    # Find R & G frequency cluster
    freq_cluster: np.ndarray = np.squeeze(  # CHANGED: torch.Tensor -> np.ndarray, torch.squeeze -> np.squeeze
        neural_gas_hist(
            samples,
            codebook_size=2,
            n_epochs=n_epochs,
            init_lr=init_lr,
            end_lr=end_lr,
            init_radius=freqs.shape[0] * rel_inti_rad,  # CHANGED: .size(dim=0) -> .shape[0]
            end_radius=end_rad,
            return_full_history=make_plots,
        )
    )  # Epochs x 2

    # Figure for plots
    fig_freq_clusts: go.Figure = None
    if make_plots:
        fig_freq_clusts = make_subplots(
            rows=1,
            cols=3,
            shared_yaxes=True,
            subplot_titles=(
                "Neural Gas convergence",
                "Normalized spectrum",
                "Actual spectrum",
            ),
        )
        fig_freq_clusts.update_layout(
            showlegend=False,
            title_text=f"Neural Gas with {n_epochs} Epochs",
            title_x=0.5,
            yaxis_title="Frequency (Hz)",
        )
        fig_freq_clusts.update_xaxes(title_text="Epochs", row=1, col=1)
        fig_freq_clusts.update_xaxes(title_text="Normalized Amplitudes", row=1, col=2)
        fig_freq_clusts.update_xaxes(title_text="Summed Amplitudes", row=1, col=3)

        # Plot green & red cluster history
        green_clust_hist: np.ndarray = freq_cluster[  # CHANGED: torch.Tensor -> np.ndarray
            :, np.argmin(freq_cluster[-1, :])  # CHANGED: torch.argmin -> np.argmin
        ]
        red_clust_hist: np.ndarray = freq_cluster[  # CHANGED: torch.Tensor -> np.ndarray
            :, np.argmax(freq_cluster[-1, :])  # CHANGED: torch.argmax -> np.argmax
        ]

        fig_freq_clusts.append_trace(
            go.Scatter(
                y=green_clust_hist,  # CHANGED: Removed .detach().cpu().numpy()
                mode="markers",
                marker={"color": "green"},
            ),
            row=1,
            col=1,
        )
        fig_freq_clusts.append_trace(
            go.Scatter(
                y=red_clust_hist,  # CHANGED: Removed .detach().cpu().numpy()
                mode="markers",
                marker={"color": "red"},
            ),
            row=1,
            col=1,
        )
        fig_freq_clusts.update_yaxes(range=(np.min(freqs), np.max(freqs)))  # CHANGED: torch.min -> np.min, torch.max -> np.max

        freq_cluster = freq_cluster[-1, :]

    red_cluster: float = float(np.max(freq_cluster))  # CHANGED: torch.max -> np.max
    green_cluster: float = float(np.min(freq_cluster))  # CHANGED: torch.min -> np.min
    blue_cluster: float = 0.0

    green_cutoff: float = (green_cluster + red_cluster) / 2
    blue_cutoff: float = green_cluster  # CHANGED: Use green_cluster instead of blue_cluster so blue covers [0, green_min]

    # Get index from frequencies:
    red_idx: np.ndarray = freqs > green_cutoff  # CHANGED: torch.Tensor -> np.ndarray
    green_idx: np.ndarray = np.logical_and(  # CHANGED: torch.Tensor -> np.ndarray, torch.logical_and -> np.logical_and
        freqs > blue_cutoff, freqs <= green_cutoff
    )
    blue_idx: np.ndarray = freqs <= blue_cutoff  # CHANGED: torch.Tensor -> np.ndarray (now covers 0 to green_min)
    # This is changed because this produces results that are matched in the paper. They claimed that their blue channel was 0 to the lower bound of green.

    # Get actual cutoff frequencies:
    blue_freq_min: float = float(np.min(freqs[blue_idx]))  # CHANGED: torch.min -> np.min
    blue_freq_max: float = float(np.max(freqs[blue_idx]))  # CHANGED: torch.max -> np.max
    red_freq_min: float = float(np.min(freqs[red_idx]))  # CHANGED: torch.min -> np.min
    red_freq_max: float = float(np.max(freqs[red_idx]))  # CHANGED: torch.max -> np.max
    green_freq_min: float = float(np.min(freqs[green_idx]))  # CHANGED: torch.min -> np.min
    green_freq_max: float = float(np.max(freqs[green_idx]))  # CHANGED: torch.max -> np.max

    print(
        f"B: {blue_freq_min} - {blue_freq_max}, "
        f"G: {green_freq_min} - {green_freq_max}, "
        f"R: {red_freq_min} - {red_freq_max}"
    )

    if make_plots:
        # Plot spectrum used for NG by final clustering (No DC)
        fig_freq_clusts.append_trace(
            go.Bar(
                x=fft_sum_no_dc[green_idx[1:]],  # CHANGED: Removed .detach().cpu().numpy()
                y=freqs[1:][green_idx[1:]],  # CHANGED: Removed .detach().cpu().numpy()
                orientation="h",
                marker_color="green",
            ),
            row=1,
            col=2,
        )
        fig_freq_clusts.append_trace(
            go.Bar(
                x=fft_sum_no_dc[red_idx[1:]],  # CHANGED: Removed .detach().cpu().numpy()
                y=freqs[1:][red_idx[1:]],  # CHANGED: Removed .detach().cpu().numpy()
                orientation="h",
                marker_color="red",
            ),
            row=1,
            col=2,
        )

        # Plot actual spectrum by final clustering
        fig_freq_clusts.append_trace(
            go.Bar(
                x=fft_sum[blue_idx],  # CHANGED: Removed .detach().cpu().numpy()
                y=freqs[blue_idx],  # CHANGED: Removed .detach().cpu().numpy()
                orientation="h",
                marker_color="blue",
            ),
            row=1,
            col=3,
        )
        fig_freq_clusts.append_trace(
            go.Bar(
                x=fft_sum[green_idx],  # CHANGED: Removed .detach().cpu().numpy()
                y=freqs[green_idx],  # CHANGED: Removed .detach().cpu().numpy()
                orientation="h",
                marker_color="green",
            ),
            row=1,
            col=3,
        )
        fig_freq_clusts.append_trace(
            go.Bar(
                x=fft_sum[red_idx],  # CHANGED: Removed .detach().cpu().numpy()
                y=freqs[red_idx],  # CHANGED: Removed .detach().cpu().numpy()
                orientation="h",
                marker_color="red",
            ),
            row=1,
            col=3,
        )

        if display_plots:
            fig_freq_clusts.show()

    img_size = list(fft.shape)  # CHANGED: .size() -> .shape
    img_size[0] = 3  # Exchange
    dynamic: np.ndarray = np.zeros(tuple(img_size), dtype=fft.dtype)  # CHANGED: torch.Tensor -> np.ndarray, torch.zeros -> np.zeros, .type_as -> dtype=
    dynamic[0, ...] = np.sum(fft[red_idx, ...], axis=0)  # CHANGED: torch.sum -> np.sum, dim -> axis
    dynamic[1, ...] = np.sum(fft[green_idx, ...], axis=0)  # CHANGED: torch.sum -> np.sum, dim -> axis
    dynamic[2, ...] = np.sum(fft[blue_idx, ...], axis=0)  # CHANGED: torch.sum -> np.sum, dim -> axis

    if make_plots:
        return dynamic, fig_freq_clusts
    else:
        # Return optimal bins instead of figure
        optimal_bins = {
            'blue': (blue_freq_min, blue_freq_max),
            'green': (green_freq_min, green_freq_max),
            'red': (red_freq_min, red_freq_max)
        }
        return dynamic, optimal_bins

def neural_gas_hist(
    samples: np.ndarray,  # CHANGED: torch.Tensor -> np.ndarray
    codebook_size: int,
    n_epochs: int,
    init_lr: float,
    end_lr: float,
    init_radius: float,
    end_radius: float,
    return_full_history: bool = False,
) -> np.ndarray:  # CHANGED: torch.Tensor -> np.ndarray
    """Neural gas clustering.

    Args:
        samples (np.ndarray): Tensor of shape nr samples x sample dimensions  # CHANGED: torch.Tensor -> np.ndarray
        codebook_size (int): Number of clusters to use.  --> "Clusters" referring to the clustering algorithm, not computational clusters. 2 here for low vs high.
        n_epochs (int): Number of epochs to run.
        init_lr (float): Initial learning rate.
        end_lr (float): End learning rate to converge to.
        init_radius (float): Initial neighborhood radius.
        end_radius (float): End neighborhood radius to converge to.
        return_full_history (bool, optional): If the cluster positions for all epochs are to be returned or just the final result. Defaults to False.

    Returns:
        np.ndarray: Array containing the cluster positions of shape Epochs X codebook_size or just codebook_size depending on return_full_history.  # CHANGED: torch.Tensor -> np.ndarray
    """
    # Norm Y values to [1e-3, 1]
    samples[:, 1] = rescale_range(samples[:, 1], new_min=1e-3, new_max=1.0)

    n_samples: int = samples.shape[0]

    codebook_vectors: np.ndarray = rescale_range(  # CHANGED: torch.Tensor -> np.ndarray
        np.random.rand(codebook_size, 1),  # CHANGED: torch.rand -> np.random.rand
        float(np.min(samples)),  # CHANGED: torch.min -> np.min
        float(np.max(samples)),  # CHANGED: torch.max -> np.max
    ).astype(samples.dtype)  # CHANGED: .type_as -> .astype

    all_cvs: list[np.ndarray] = [codebook_vectors]  # CHANGED: torch.Tensor -> np.ndarray

    for epoch in range(n_epochs):
        ep_power: float = epoch / n_epochs
        learning_rate: float = init_lr * (end_lr / init_lr) ** ep_power
        neighborhood_radius: float = (
            init_radius * (end_radius / init_radius) ** ep_power
        )
        indexes: np.ndarray = np.random.permutation(n_samples)  # CHANGED: torch.Tensor -> np.ndarray, torch.randperm -> np.random.permutation

        for index in indexes:
            sample: np.ndarray = samples[index]  # CHANGED: torch.Tensor -> np.ndarray

            # Euclidean distances between sample and each codebook vector
            distances: np.ndarray = np.linalg.norm(  # CHANGED: torch.Tensor -> np.ndarray, torch.norm -> np.linalg.norm
                sample[0] - codebook_vectors, axis=1  # CHANGED: dim -> axis
            )  # codebook_size x n_features

            # Compute the rank of each codebook vector
            ranks: np.ndarray = np.argsort(distances)  # CHANGED: torch.Tensor -> np.ndarray, torch.argsort -> np.argsort

            # Update all codebook vectors
            # Amplitude is multiplied as a weight
            codebook_vectors = codebook_vectors + learning_rate * sample[1] * np.exp(  # CHANGED: torch.exp -> np.exp
                -ranks / neighborhood_radius
            )[:, None] * (sample[0] - codebook_vectors)

        if return_full_history:
            all_cvs.append(codebook_vectors)

    if return_full_history:
        return np.stack(all_cvs)  # CHANGED: torch.stack -> np.stack
    else:
        return codebook_vectors


def rgb_frequency_binning(
    DOCTData: DOCTData,
    in_file: pathlib.Path,
    out_folder: pathlib.Path,
    fps: float,
    linearize: bool = True,
):
    """Default RGB frequency binning.

    Args:
        in_file (pathlib.Path): tiff file containing registered data of shape TxHxW.
        out_folder (pathlib.Path): Folder to save results into.
        fps (float): Frames per second of the in_file data. Though, this is just for plotting purposes and can otherwise arbitrarily be chosen if unsure.
        linearize (bool, optional): If in_file data is logarithmic. Defaults to True.
    """
    # Load file
    if in_file is not None:
        img: np.ndarray = tifffile.imread(in_file).astype(np.float32)  # CHANGED: torch.Tensor -> np.ndarray, removed torch.from_numpy
    else:
        img: np.ndarray = DOCTData.data.astype(np.float32)  # CHANGED: torch.Tensor -> np.ndarray, removed torch.from_numpy

    # Linearize dB data if needed
    if linearize:
        img = 10 ** (img / 20)

    time_res: float = 1 / fps

    # Logarithmic standard deviation per pixel
    std: np.ndarray = np.log10(np.std(img, axis=0))  # CHANGED: torch.Tensor -> np.ndarray, torch.log10 -> np.log10, torch.std -> np.std, dim -> axis

    # FFT over time axis (for each pixel) and discard negative spectrum
    nr_frames: int = img.shape[0]  # CHANGED: .size(0) -> .shape[0]
    pos_side: int = nr_frames // 2 + 1
    fft: np.ndarray = np.abs(np.fft.fft(img, axis=0))[:pos_side, ...]  # CHANGED: torch.Tensor -> np.ndarray, torch.abs -> np.abs, torch.fft.fft -> np.fft.fft, dim -> axis
    freqs: np.ndarray = np.abs(  # CHANGED: torch.Tensor -> np.ndarray, torch.abs -> np.abs
        np.fft.fftfreq(nr_frames, time_res)[:pos_side, ...]  # CHANGED: torch.fft.fftfreq -> np.fft.fftfreq
    )

    # Bin frequencies to RGB
    dynamic: np.ndarray  # CHANGED: torch.Tensor -> np.ndarray
    result: dict | go.Figure
    dynamic, result = rgb_by_neural_gas_b_is_dc_hist(
        fft,
        freqs,
        n_epochs=10,
        init_lr=0.1,
        rel_inti_rad=0.1,
        make_plots=False,
        display_plots=False,
    )
    # Return optimal bins instead of saving images
    if isinstance(result, dict):
        return result

    # Skip image processing, saving and histogram matching
    # std = normalize_and_expand_channels(std)
    # dynamic = hist_matching(dynamic, std)
    # save_path: pathlib.Path = out_folder.joinpath(f"{in_file.stem}-dyn-NG.png")
    # save_uint8(dynamic, save_path)


def check_mkdir(directory_or_file: pathlib.Path, exists_ok: bool = True) -> None:
    """Create a directory for the give path if none exists.

    Args:
        directory_or_file (pathlib.Path): Path to an existing or new directory.
    """
    directory: pathlib.Path = get_directory(directory_or_file)
    directory.mkdir(parents=False, exist_ok=exists_ok)


def get_directory(directory_or_file: pathlib.Path) -> pathlib.Path:
    """Get the directory of a path.

    Args:
        directory_or_file (pathlib.Path): Path to either a file or directory

    Returns:
        pathlib.Path: Directory of the path.
    """
    directory: pathlib.Path

    if directory_or_file.is_file():
        directory = directory_or_file.parent
    else:
        directory = directory_or_file

    return directory


def rescale_range(
    in_arr: np.ndarray, new_min: float = 0.0, new_max: float = 1.0  # CHANGED: torch.Tensor -> np.ndarray
) -> np.ndarray:  # CHANGED: torch.Tensor -> np.ndarray
    """Rescale the range of a given array to the new min and max.

    Args:
        in_arr (np.ndarray): Input array to rescale the range of.  # CHANGED: torch.Tensor -> np.ndarray
        new_min (float): New minimum. Defaults to 0.
        new_max (float): New maximum. Defaults to 1.

    Returns:
        np.ndarray: Range rescaled array.  # CHANGED: torch.Tensor -> np.ndarray
    """
    old_min: float = float(np.min(in_arr))  # CHANGED: torch.min -> np.min
    old_max: float = float(np.max(in_arr))  # CHANGED: torch.max -> np.max

    in_arr = (in_arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    return in_arr


def normalize_and_expand_channels(arr: np.ndarray, dims: int = 3) -> np.ndarray:  # CHANGED: torch.Tensor -> np.ndarray (both places)
    """Normalizes input to [0, 1] and multiplies it into dims channels.

    Args:
        arr (np.ndarray): Input to normalize and expand  # CHANGED: torch.Tensor -> np.ndarray
        dims (int, optional): Amount of expansions. Defaults to 3.

    Returns:
        np.ndarray: Normalized tensor of shape dims x H x W.  # CHANGED: torch.Tensor -> np.ndarray
    """
    # Normalize to [0, 1]
    arr = rescale_range(arr)

    # Expand to CxHxW
    arr_channels: np.ndarray = np.repeat(arr[None, ...], dims, axis=0)  # CHANGED: torch.Tensor -> np.ndarray, torch.repeat_interleave -> np.repeat, dim -> axis

    return arr_channels


def saturate_values(
    img: np.ndarray, clip_top: float = 0.9999, clip_bottom: float = 0.001  # CHANGED: torch.Tensor -> np.ndarray
) -> np.ndarray:  # CHANGED: torch.Tensor -> np.ndarray
    """Clips values of an input to the given upper and lower percentages.

    Args:
        img (np.ndarray): Input tensor to clip.  # CHANGED: torch.Tensor -> np.ndarray
        clip_top (float, optional): Upper clip limit. Defaults to 0.9999 (0.01 %).
        clip_bottom (float, optional): Lower clip limit. Defaults to 0.001 (0.1 %).

    Returns:
        np.ndarray: Clipped Array.  # CHANGED: torch.Tensor -> np.ndarray
    """
    unique_vals: np.ndarray = np.unique(img)  # CHANGED: torch.Tensor -> np.ndarray, torch.unique -> np.unique

    # Clip top values
    sat_limit = float(np.max(unique_vals[: int(unique_vals.shape[0] * clip_top)]))  # CHANGED: torch.max -> np.max, .size(0) -> .shape[0]
    img[img > sat_limit] = sat_limit

    # Clip bottom values
    sat_limit = float(np.min(unique_vals[int(unique_vals.shape[0] * clip_bottom) :]))  # CHANGED: torch.min -> np.min, .size(0) -> .shape[0]
    img[img < sat_limit] = sat_limit

    return img


def hist_matching(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:  # CHANGED: torch.Tensor -> np.ndarray (all 3 places)
    """Channel wise histogram matching of img1 onto the histogram of img2.

    Args:
        img1 (np.ndarray): Array whose histogram is to be adjusted.  # CHANGED: torch.Tensor -> np.ndarray
        img2 (np.ndarray): Array to use as target histogram  # CHANGED: torch.Tensor -> np.ndarray

    Returns:
        np.ndarray: Histogram matched output  # CHANGED: torch.Tensor -> np.ndarray
    """
    # This is 1 for all non-empty channels and 0 for all empty channels.
    max_vals: np.ndarray = np.max(img1.reshape(img1.shape[0], -1), axis=-1)  # CHANGED: torch.Tensor -> np.ndarray, torch.max -> np.max, .size(0) -> .shape[0], dim -> axis, removed [0] index

    out_img: np.ndarray = match_histograms(  # CHANGED: torch.Tensor -> np.ndarray, removed torch.from_numpy and .type_as
        img1, img2, channel_axis=0  # CHANGED: Removed .detach().cpu().numpy() calls
    ).astype(img1.dtype)  # CHANGED: Added .astype for type matching

    # Zero previously empty channels again (they get set to 1 by match_histograms)
    while max_vals.ndim < out_img.ndim:
        max_vals = np.expand_dims(max_vals, axis=-1)  # CHANGED: torch.unsqueeze -> np.expand_dims, dim -> axis
    out_img *= max_vals

    return out_img


def save_uint8(img: np.ndarray, path: pathlib.Path) -> None:  # CHANGED: torch.Tensor -> np.ndarray
    """Save input as uint8 file to the given path.

    Args:
        img (np.ndarray): [0, 1] normalized image.  # CHANGED: torch.Tensor -> np.ndarray
        path (pathlib.Path): Path to save to, including the suffix.
    """
    # CHANGED: Replaced torchvision.utils.save_image with matplotlib/PIL approach
    from PIL import Image
    # Convert from CxHxW to HxWxC and scale to 0-255
    img_uint8 = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)  # CHANGED: Added conversion
    Image.fromarray(img_uint8).save(path)  # CHANGED: Save using PIL


def neural_gas_clustering(DOCTData, frame_rate=100, n_epochs=10, init_lr=1.0, end_lr=1e-3, 
                          rel_inti_rad=0.1, end_rad=0.01, make_plots=False, display_plots=False,
                          save_plot_path=None):
    """
    Perform neural gas clustering to automatically discover optimal frequency bins for RGB.
    
    Uses the neural gas algorithm to find natural clusters in the frequency spectrum.
    Returns the discovered frequency bins in RGB order.
    
    Parameters
    ----------
    DOCTData : DOCTData
        Input DOCTData object (should be in linear scale)
    frame_rate : float, default 100
        Frame rate in Hz
    n_epochs : int, default 10
        Number of epochs for neural gas clustering
    init_lr : float, default 1.0
        Initial learning rate for neural gas clustering
    end_lr : float, default 1e-3
        Final learning rate for neural gas clustering
    rel_inti_rad : float, default 0.5
        Initial relative neighborhood radius for neural gas clustering
    end_rad : float, default 0.01
        Final neighborhood radius for neural gas clustering
    make_plots : bool, default False
        If True, generate diagnostic plots showing clustering convergence
    display_plots : bool, default False
        If True and make_plots=True, display the plots interactively
    save_plot_path : str or Path, optional
        If provided and make_plots=True, save the plot to this path (e.g., 'clustering.html' or 'clustering.jpg')
        
    Returns
    -------
    list of tuples
        Frequency bins in RGB order: [(red_min, red_max), (green_min, green_max), (blue_min, blue_max)]
    dict (optional, if make_plots=True)
        Dictionary containing diagnostic information:
        - 'figure': plotly figure object
        - 'green_cluster': green cluster center frequency
        - 'red_cluster': red cluster center frequency
        - 'green_cutoff': cutoff between green and red
    """
    # Compute FFT
    I = DOCTData.data - np.mean(DOCTData.data, axis=0)
    fft_result = np.fft.fft(I, axis=0)
    fft_magnitude = np.abs(fft_result)
    fft_freqs = np.fft.fftfreq(I.shape[0], d=1/frame_rate)
    
    # Keep only positive frequencies (INCLUDING 0 Hz DC component, as expected by original algorithm)
    # Match torch implementation: take positive half and abs() to ensure 0.0 (not -0.0)
    pos_side = I.shape[0] // 2 + 1
    fft_positive = fft_magnitude[:pos_side]
    freqs_positive = np.abs(fft_freqs[:pos_side])  # abs() to match torch behavior
    
    # Run neural gas clustering
    _, result = rgb_by_neural_gas_b_is_dc_hist(
        fft=fft_positive,
        freqs=freqs_positive,
        n_epochs=n_epochs,
        init_lr=init_lr,
        end_lr=end_lr,
        rel_inti_rad=rel_inti_rad,
        end_rad=end_rad,
        make_plots=make_plots,
        display_plots=display_plots
    )
    
    # Extract RGB bins
    if make_plots:
        # result is a plotly figure
        fig = result
        # We need to get the bins from the figure or recompute - let's recompute without plots
        _, optimal_bins = rgb_by_neural_gas_b_is_dc_hist(
            fft=fft_positive,
            freqs=freqs_positive,
            n_epochs=n_epochs,
            init_lr=init_lr,
            end_lr=end_lr,
            rel_inti_rad=rel_inti_rad,
            end_rad=end_rad,
            make_plots=False,
            display_plots=False
        )
        
        # Save plot if path provided
        if save_plot_path:
            import pathlib
            save_path = pathlib.Path(save_plot_path)
            if save_path.suffix in ['.html', '.htm']:
                fig.write_html(save_plot_path)
                print(f"Saved clustering plot to {save_plot_path}")
            else:
                fig.write_image(save_plot_path, scale=3.0)
                print(f"Saved clustering plot to {save_plot_path}")
    else:
        # result is optimal_bins dict
        optimal_bins = result
    
    # Convert bins dict to RGB order (red, green, blue)
    RGB = [
        tuple(optimal_bins['red']),
        tuple(optimal_bins['green']),
        tuple(optimal_bins['blue'])
    ]
    
    # Return RGB bins and optionally diagnostic info
    if make_plots:
        diagnostics = {
            'figure': fig,
            'blue_range': optimal_bins['blue'],
            'green_range': optimal_bins['green'],
            'red_range': optimal_bins['red'],
        }
        return RGB, diagnostics
    
    return RGB
