# dynamic-oct

A scalable, object-oriented, and modular analysis toolkit for dynamic OCT data.

Dynamic OCT leverages signal intensity fluctuations to identify unique signatures of cellular activity and tissue-specific dynamics. This repository provides a local Python package, `doct`, for loading time-series OCT data, preprocessing it, running published dynamic OCT analyses, and storing outputs in a shared `DOCTData` container.

## Installation

This project is intended to be installed locally.

### Conda setup

```bash
conda env create -f environment.yml
conda activate dynamic-oct
```

The environment file installs the package in editable mode with `pip install -e .`, so local code changes are immediately available inside the environment.

### Alternative: existing environment

If you already have a Python or conda environment you want to use, go to it and run:

```bash
pip install -e .
```

## Basic usage

```python
# Import statements for full functionality
import doct
from doct import DOCTData
from doct import readwrite, preprocessing, visual
from doct.analysis import core
from doct.analysis import motility as motility_module
from doct.analysis import neural_gas, aLIV_swift

ddata = read_tiffs("path/to/tiff_folder") # Assuming log scaled data
liv(ddata)
show_heatmap(ddata.results["liv"])
```

## Package layout

- `doct/DOCTData.py`: initializes the shared container object for raw data, metadata, and analysis results
- `doct/readwrite.py`: TIFF import/export helpers
- `doct/preprocessing.py`: masking, cropping, scaling, and trimming
- `doct/visual.py`: visualization helpers
- `doct/analysis/`: dynamic OCT analysis methods, including all that is in the Implemented methods section
- `notebooks/tutorial.ipynb`: interactive usage notebook

## Implemented methods

- Standard deviation
- Logarithmic intensity variance (LIV)
- OCT correlation decay speed
- RGB frequency binning
- Neural gas clustering for RGB frequency binning
- Authentic LIV (aLIV) and swiftness
- Motility metrics ($M$, $\alpha$, $R^2$)

## Notes on data shape

The expected input shape is `(T, H, W)` for a time series of 2D OCT frames. Windowed processing can be used to analyze larger concatenated acquisitions.

## Source material

The package adapts and refactors methods from:

1. https://github.com/ComputationalOpticsGroup/COG-dynamic-OCT-contrast-generation-library
2. https://github.com/noahheldt/A-guide-to-dynamic-OCT-data-analysis

## Contact
**Joesph Beller** - joesph.beller@bcm.edu \
Please use the [GitHub Issues](https://github.com/LarinaLab/dynamic-oct/issues) page for questions, bug reports, and feature requests.
