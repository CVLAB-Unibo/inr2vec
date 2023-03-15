# Deep Learning on INRs of Shapes

Official code for the paper "_Deep Learning on Implicit Neural Representations of Shapes_", published 
at ICLR 2023.  
Authors: Luca De Luigi*, Adriano Cardace*, Riccardo Spezialetti*, Pierluigi Zama Ramirez, Samuele 
Salti, Luigi Di Stefano.  
\* joint first authorship

---
The code contained in this repository has been tested on Ubuntu 20.04 with Python 3.8.6.

## Setup
Create a virtual environment and install the library `pycarus`:
```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -U pip setuptools
$ pip install pycarus
```
Then, try to import `pycarus` to get the command that you can run to install all the needed Pytorch libraries:
```
$ python3
>>> import pycarus
...
ModuleNotFoundError: PyTorch is not installed. Install it by running: source /XXX/.venv/lib/python3.8/site-packages/pycarus/install_torch.sh
```
In this example, you can install all the needed Pytorch libraries by running:
```
$ source /XXX/.venv/lib/python3.8/site-packages/pycarus/install_torch.sh
```
This script downloads and installs the wheels for torch, torchvision, pytorch3d and torch-geometric.
Occasionally, it may fails due to pytorch3d wheel not being available anymore. If that happens,
please let us know or try to install pytorch3d manually.  
Finally install the other dependencies:
```
$ pip install hesiod torchmetrics wandb h5py==3.0.0
```
## Experiments
The code for each experiment has been organized in a separate directory, containing also a README file with all the instructions.  

## Datasets
Please contact us if you need access to the datasets used in the experiments, both the ones containing the raw 3D shapes and the ones with the INRs.

## Cite us
If you find our work useful, please cite us:
```
@inproceedings{deluigi2023inr2vec,
    title = {Deep Learning on Implicit Neural Representations of Shapes},
    author = {Luca De Luigi 
              and Adriano Cardace 
              and Riccardo Spezialetti 
              and Pierluigi Zama Ramirez 
              and Samuele Salti 
              and Luigi Di Stefano},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year = {2023}
}
```
