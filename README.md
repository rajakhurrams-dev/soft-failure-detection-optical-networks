# Soft-failure detection in optical fibre networks

Code for the MSc Data Science dissertation *Machine Learning-Based Detection and Early Identification of Soft-Failures in Optical Fiber Networks* (University of East London / UNICAF, module UEL-DS-7010).

Author: Raja Khurram Sajjad Khan

## Contents

| File | Purpose |
|---|---|
| `soft_failure_detection.ipynb` | Executed notebook with all outputs saved. Every number, table and figure in the dissertation comes from it. |
| `soft_failure_detection.py` | The same code as a plain Python script (jupytext percent format). |
| `requirements.txt` | Package versions used for the reported run (Python 3.13). |

## Data

The dataset is not included here. The code uses the public optical failure dataset of the InRete Lab, Scuola Superiore Sant'Anna:
https://github.com/Network-And-Services/optical-failure-dataset (commit `7f0e9c0`, 16 February 2022).

The notebook clones that repository automatically if it is not found at `DATA_DIR` (default `../../data/optical-failure-dataset`, set at the top of the notebook).

## How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 soft_failure_detection.ipynb
```

Or run the script directly: `python soft_failure_detection.py`.

Set the environment variable `SFD_FAST=1` for a quick smoke test with reduced hyperparameter grids (about 8 minutes on 2 CPU cores). The reported results use the full run.

Outputs are written to `../figures` (17 figures) and `../tables` (42 CSV tables), relative to the folder the code is run from.
