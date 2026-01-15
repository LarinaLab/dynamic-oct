import numpy as np
import os
import json
import imageio
from natsort import natsorted
from tqdm import tqdm
from DOCTData import DOCTData


def read_tiffs(path: str, endpoint = False) -> DOCTData:
    """
    Load DOCT data from a folder of numbered TIFF files.
    
    Parameters
    ----------
    path : str
        Path to folder containing TIFF files, with a numerated suffix
    endpoint
        If true, stops reading in the files at a certain number (helpful for if you already anticipate trimming)
    Returns
    -------
    DOCTData
        DOCTData object with loaded image stack
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")
    
    if not os.path.isdir(path):
        raise ValueError(f"Path {path} is not a directory.")
    
    filenames = natsorted([f for f in os.listdir(path) 
                           if f.endswith(".tiff") or f.endswith(".tif")])
    
    if not filenames:
        raise ValueError(f"No TIFF files found in {path}")
    
    images = []
    for fname in tqdm(filenames, desc="Loading TIFF images", unit="image"):
        frame = imageio.imread(os.path.join(path, fname))
        images.append(frame)
        if endpoint and len(images) == endpoint:
            break
    
    image_stack = np.stack(images, axis=0)
    
    return DOCTData(
        data=image_stack,
        meta={'source_path': path, 'n_frames': len(images)}
    )

def save_tiff_stack(array: np.ndarray, path: str, array_name: str = None) -> None:
    """
    Save a 3D numpy array as a multi-frame TIFF file.
    
    Parameters
    ----------
    array : np.ndarray
        3D array to save (frames x height x width)
    path : str
        Directory path or full file path to save the TIFF file
    array_name : str, optional
        Name to use for the file (e.g., 'std_3d'). If not provided and path is a directory,
        defaults to 'output'
        
    """
    if array.ndim != 3:
        raise ValueError("Input array must be 3D (frames x height x width).")
    
    # Ensure path has .tif extension
    if not path.endswith(('.tif', '.tiff')):
        # If array_name provided, use it; otherwise use 'output'
        name = array_name if array_name else 'output'
        path = f"{path}/{name}.tif"
    
    # Check for negative values or values requiring additional precision
    if np.any(array < 0) or np.any(array % 1 != 0) or np.any(array > np.iinfo(np.uint16).max):
        imageio.mimwrite(path, array.astype(np.float32))
        print(f"Saved TIFF stack to {path} (as float32 due to negative values or additional precision)")
    else:
        imageio.mimwrite(path, array.astype(np.uint16))
        print(f"Saved TIFF stack to {path} (as uint16)")

def read_tiff_stack(path: str) -> DOCTData:
    """
    Load DOCT data from a single multi-frame TIFF file.
    
    Parameters
    ----------
    path : str
        Path to multi-frame TIFF file
        
    Returns
    -------
    DOCTData
        DOCTData object with loaded image stack
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    
    with imageio.get_reader(path) as reader:
        frames = []
        n_frames = reader.count_frames()
        for frame in tqdm(reader, desc="Loading TIFF frames", 
                         unit="frame", total=n_frames):
            frames.append(frame)
        image_stack = np.stack(frames, axis=0)
    
    return DOCTData(
        data=image_stack,
        meta={'source_path': path, 'n_frames': len(frames)}
    )


def save_DOCTData(DOCTData, path: str = None) -> None:
    """
    Save DOCTData object to disk.
    
    Saves data as compressed .npz file and metadata/results as .json file.
    
    Parameters
    ----------
    doct_obj : DOCTData
        DOCTData object to save
    path : str
        Path to save files (without extension). Will create path.npz and path.json
        
    """
    if path is None:
        if 'source_path' in DOCTData.meta:
            path = DOCTData.meta['source_path']
        else:
            raise ValueError("No path provided and 'source_path' not in DOCTData meta.")

    arrays_to_save = {'data': DOCTData.data}
    for key, value in DOCTData.meta.items():
        if isinstance(value, np.ndarray):
            arrays_to_save[f'meta_{key}'] = value
    for key, value in DOCTData.results.items():
        if isinstance(value, np.ndarray):
            arrays_to_save[f'results_{key}'] = value

    np.savez_compressed(f'{path}.npz', **arrays_to_save)
    
    # Save non-array metadata/results to .json
    save_dict = {
        'meta': {k: v for k, v in DOCTData.meta.items() if not isinstance(v, np.ndarray)},
        'results': {k: v for k, v in DOCTData.results.items() if not isinstance(v, np.ndarray)}
    }
    
    with open(f'{path}.json', 'w') as f:
        json.dump(save_dict, f, indent=2, default=str)
    
    print(f"Saved DOCTData to {path}.npz and {path}.json")


def load_DOCTData(path: str) -> DOCTData:
    """
    Load DOCTData object from disk.
    
    Parameters
    ----------
    path : str
        Path to saved files (without extension). Will load from path.npz and path.json
        
    Returns
    -------
    DOCTData
        Loaded DOCTData object
        
    """
    if not os.path.exists(f'{path}.npz'):
        raise FileNotFoundError(f"Could not find {path}.npz")
    
    arrays = np.load(f'{path}.npz', allow_pickle=True)
    data = arrays['data']
    
    # Load metadata from JSON
    meta = {}
    results = {}
    if os.path.exists(f'{path}.json'):
        with open(f'{path}.json', 'r') as f:
            saved_dict = json.load(f)
            meta = saved_dict.get('meta', {})
            results = saved_dict.get('results', {})
    
    # Add array metadata and results back from .npz
    for key in arrays.files:
        if key.startswith('meta_'):
            meta_name = key[5:]  # Remove 'meta_' prefix
            meta[meta_name] = arrays[key]
        elif key.startswith('results_'):
            result_name = key[8:]  # Remove 'results_' prefix
            results[result_name] = arrays[key]
    
    print(f"Loaded DOCTData from {path}.npz and {path}.json")
    
    return DOCTData(data=data, meta=meta, results=results)