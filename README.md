# Adaptive Bernstein Change Detector for High-Dimensional Data Streams 

This repository contains code for the paper "Adaptive Bernstein Change Detector for High-Dimensional Data Streams" currently under review at ICDE 2023.

## Installation of dependencies

We have exported our conda environments into a `requirements.yml` file (used on our linux-server) and a `requirements-windows.yml` file (used on windows PCs). To recreate the environments, run 
- `conda env create -f requirements.yml` or 
- `conda env create -f requirements-windows.yml`.

If you encounter any problems during installation please have a look at the packages listed in the file and install them manually.

## Related repositories

Please add the following repositories to this project prior to executing any experiments:

### Data stream generators
- https://github.com/heymarco/StreamGenerators
- Add to this repository via `python -m pip install -e git+https://github.com/heymarco/StreamDatasets.git#egg=stream-datasets --upgrade`

### Change detectors
- https://github.com/heymarco/ChangeDetectors
- Add to this repository via `python -m pip install -e git+https://github.com/heymarco/ChangeDetectors.git#egg=change-detectors --upgrade`

## File structure

- To recreate the experiments in the paper, run any of the files starting with `E_` (for experiments) 
- To recreate the plots run any of the `mining/M_` files (for experiment mining). 
