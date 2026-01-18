# dynamic-oct

A scalable and object-oriented analysis toolkit for analyzing dynamic OCT data.

Dynamic OCT leverages signal intensity fluctuations to find unique signatures of cellular activities and tissues. This repository contains code to compute dynamic OCT results and can be found in the `doct/` folder.

## Coming Soon

Tutorials to use this framework on your own data will be pushed to this repository soon! The preferred data type is **T×H×W** or **TxWxH**, but volumetric "blocks" could be concatenated together, and `apply_windowed()` in `tools.py` could be applied to process these data. Current 4D data from the Larina Lab that have been processed using `apply_windowed()` were acquired with a slow volumetric scanning protocol, where we collected ~128 frames per pixel of resolution (i.e., 10,000 B-scans covering ~95µm).

## Implemented Methods

The following published methods have been implemented:
- Standard deviation
- Logarithmic intensity variance (LIV)
- OCT correlation decay speed
- RGB frequency binning 
- Neural gas clustering for RGB frequency binning [2]
- Authentic LIV (aLIV) and Swiftness [1,2]
- Motility metrics ($M$, $\alpha$, $R^2$) [2]


### Original code repositories which were adapted to fit into this object-oriented framework

1. https://github.com/ComputationalOpticsGroup/COG-dynamic-OCT-contrast-generation-library

2. https://github.com/noahheldt/A-guide-to-dynamic-OCT-data-analysis

