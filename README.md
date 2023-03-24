![pytomoatt](https://user-images.githubusercontent.com/7437523/222756855-9a178426-a4b1-4522-9022-8db4b017e7d8.png)

## Python API for TomoATT, a package for Eikonal-based seismic tomography


[![Python Package using Conda](https://github.com/MIGG-NTU/PyTomoATT/actions/workflows/build-test-conda.yml/badge.svg?branch=devel)](https://github.com/MIGG-NTU/PyTomoATT/actions/workflows/build-test-conda.yml)
[![Build documentations](https://github.com/MIGG-NTU/PyTomoATT/actions/workflows/build-docs.yml/badge.svg?branch=docs)](https://migg-ntu.github.io/PyTomoATT/)
[![codecov](https://codecov.io/gh/MIGG-NTU/PyTomoATT/branch/devel/graph/badge.svg?token=EYOV0WOA2Y)](https://codecov.io/gh/MIGG-NTU/PyTomoATT)

![PyPI - License](https://img.shields.io/pypi/l/pytomoatt)
![PyPI](https://img.shields.io/pypi/v/pytomoatt)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytomoatt)


PyTomoATT is a Python module that provides an interface for TomoATT, a software package for seismic tomography analysis. With PyTomoATT, users can create TomoATT projects, manage input data, and generate 3D initial models, 3D checkerboards, and slices of horizontal and vertical cross sections.

### Authors
- [Mijian Xu](https://xumijian.me)
- [Masaru Nagaso](https://mnagaso.github.io)

### Inclusion

- Processing for the input travel time data.
- Create initial model from CRUST1.0 and custom models.
- Add checkerboard on an exists model.
- API for output data of TomoATT (kernel, travel time field ...).
- Post-processing for final model.
- Interpolation of map views and cross-sections for model and output data.

## Installation

PyTomoATT can be installed quickly via the PyPI with a command as following:

```
pip install pytomoatt
```

See section [Installation](https://migg-ntu.github.io/PyTomoATT/installation.html) in online documentation for more details.


## Documentation

The complete API documentation, and tutorials can be accessed at [PyTomoATT Documentation](https://migg-ntu.github.io/PyTomoATT/index.html).
