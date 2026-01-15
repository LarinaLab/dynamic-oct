from typing import Dict, Any, Optional
import numpy as np
import copy as _copy


class DOCTData:
    """
    A simple container for DOCT image timeseries data.
    
    Attributes
    ----------
    data : np.ndarray or None
        The image timeseries array, shape (T, W, H) or (T, H, W)
    meta : dict
        Metadata dictionary for storing acquisition parameters, timestamps, etc.
    results : dict
        Dictionary for storing analysis results (filtered data, masks, calculations, etc.)
    """
    
    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
        results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize DOCTData object.
        
        Parameters
        ----------
        data : np.ndarray, optional
            Image timeseries array
        meta : dict, optional
            Metadata dictionary
        results : dict, optional
            Results dictionary
        """
        self.data = data if data is None else np.asarray(data)
        self.meta = meta if meta is not None else {}
        self.results = results if results is not None else {}
    
    def __repr__(self) -> str:
        """Return string representation."""
        shape_str = str(self.data.shape) if self.data is not None else "None"
        n_meta = len(self.meta)
        n_results = len(self.results)
        return (
            f"DOCTData(shape={shape_str}, "
            f"meta_keys={n_meta}, results_keys={n_results})"
        )
    
    def copy(self, deep: bool = True) -> 'DOCTData':
        """Return a copy of the object.
        
        Parameters
        ----------
        deep : bool, default True
            If True, deep copy all arrays and dicts.
            If False, shallow copy (data arrays are shared).
        """
        if deep:
            return _copy.deepcopy(self)
        else:
            return DOCTData(
                data=self.data,
                meta=self.meta.copy(),
                results=self.results.copy(),
            )
    
    @property
    def shape(self) -> tuple:
        """Shape of the data array."""
        return self.data.shape if self.data is not None else (0,)
    
    @property
    def n_frames(self) -> int:
        """Number of frames (first dimension)."""
        return self.data.shape[0] if self.data is not None else 0