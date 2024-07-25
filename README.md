# Adaptive Bernstein Change Detector for High-Dimensional Data Streams 

This repository contains code for the paper "Adaptive Bernstein Change Detector for High-Dimensional Data Streams" ([https://arxiv.org/abs/2306.12974](https://arxiv.org/abs/2306.12974)).

## Abstract
Change detection is of fundamental importance when analyzing data streams. Detecting changes both quickly and accurately enables monitoring and prediction systems to react, e.g., by issuing an alarm or by updating a learning algorithm. However, detecting changes is challenging when observations are high-dimensional.

In high-dimensional data, change detectors should not only be able to identify when changes happen, but also in which subspace they occur. Ideally, one should also quantify how severe they are. Our approach, ABCD, has these properties. ABCD learns an encoder-decoder model and monitors its accuracy over a window of adaptive size. ABCD derives a change score based on Bernstein’s inequality to detect deviations in terms of accuracy, which indicate changes.

Our experiments demonstrate that ABCD outperforms its best competitor by up to 20 % in F1-score on average. It can also accurately estimate changes' subspace, together with a severity measure
that correlates with the ground truth.

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

## Execution

- `conda activate abcd`
- To recreate the experiments in the paper, run any of the files starting with `E_` (for experiments)
  - The code will run in parallel by default on *all - 1* available cores. 
- To recreate the plots run any of the `mining/M_` files (for experiment mining). 
