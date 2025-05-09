# MicrostructureOrientation

A python module for the analysis of microstructure orientation  of transparency 
images of soft tissue, and the use of a Kelvin-based mechanical model to
describe their mechanical behavior.

## Requirements

To install the module, you'll need:

* Python 3.12 exactly (the GPU-related packages are not available for Python 
  3.13 yet, 09/05/2025)
* git
* A C++ compiler (g++ 11.4.0 on my machine)
* The eigen C++ library (libeigen3-dev version 3.4.0 from apt on my machine)

## Installation

Run the following commands to install the module:

    git clone https://github.com/WeisLeDocto/MicrostructureOrientation.git
    cd MicrostructureOrientation
    python3.12 -m venv venv
    venv/bin/python -m pip install --upgrade pip
    venv/bin/python -m pip install .

## Usage

You can get inspiration from the files in the `examples` folder to use the
module on your own data. For instance, run:

    venv/bin/python examples/optimize_diagonals.py
