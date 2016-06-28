[![Build](https://img.shields.io/travis/DarkEnergySurvey/ugali.svg)](https://travis-ci.org/DarkEnergySurvey/ugali)
[![Release](https://img.shields.io/github/tag/DarkEnergySurvey/ugali.svg)](../../releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../../)

Overview
--------

The ultra-faint galaxy likelihood (UGaLi) toolkit provides a set of python classes and functions developed for maximum-likelihood-based studies of Milky Way satellite galaxies. The primary inputs are stellar object catalogs derived from optical photometric surveys and the coverage masks of those surveys.

[Keith Bechtol](https://github.com/bechtol) & [Alex Drlica-Wagner](https://github.com/kadrlica)

Installation
------------

There are several ways to install `ugali` and it's complimentary isochrone library.

The easiest way to install `ugali` is through [PyPi](https://pypi.python.org/pypi) using `pip`:
```
# To install just the source code
pip install ugali

# To install source code and isochrone library
pip install ugali --install-option "--isochrones"

# To specify a path for the isochrone library
pip install ugali --install-option "--isochrones" --install-option "--isochrones-path <PATH>"
```
The isochrone library is a ~100 MB tarball. The default isochrone installation location is set by the `UGALIDIR` environment variable and defaults to `$HOME/.ugali`. The download and unpacking of the isochrone files might make it appear that your `pip` installation has stalled. Unfortunately, `pip` [will not display a progress bar](https://github.com/pypa/pip/issues/2732#issuecomment-97119093) during this delay.

To get the most up-to-date version of `ugali`, you can download the source code from github and install it using `setup.py`:
```
# Clone source code from the parent repository
git clone https://github.com/DarkEnergySurvey/ugali.git
cd ugali

# Install the python source code
python setup.py install

# Also install the isochrone library
python setup.py install --isochrones

# Specify the isochrone install path
python setup.py install --isochrones --isochrones-path=<PATH>
```

The code uses the `UGALIDIR` environment variable to find the isochrones (defaults to `$HOME/.ugali`). If you install the isochrones in a non-standard location be sure to set `UGALIDIR` so `ugali` can find them:
```
export UGALIDIR=$<PATH>/isochrones
```

Usage Examples
--------------
Examples go here...

Dependencies
------------
These should mostly be taken care of by PyPi with a `pip install`.

### Python packages:
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](http://matplotlib.org/)
* [pyfits](http://www.stsci.edu/institute/software_hardware/pyfits)
* [healpy](https://github.com/healpy/healpy)
* [astropy](http://www.astropy.org/)
* [emcee](http://dan.iel.fm/emcee/current/)
* [pyyaml](http://pyyaml.org/)

### Mangle:
Not a strict dependency. Used to interface with masks produced by
the Dark Energy Survey Data Mangement group. Download and documentation 
available at http://space.mit.edu/~molly/mangle/

### Isochrones:
The `ugali` uses a library of stellar isochrones packaged with `ugali` releases. These isochrones come dominantly from two different groups:
* Padova isochrones (http://stev.oapd.inaf.it/cgi-bin/cmd)
* Dartmouth isochrones (http://stellar.dartmouth.edu/models/isolf_new.html)

Conventions
-----------

### Indexing:
array[index_z][index_y][index_x]

### Naming:
* package_name
* module_name.py
* ClassName
* functionName
* variable_name

ABBREVIATIONS
-------------
* IMF: initial mass function
* CMD: color-magnitude diagram
* ROI: region of interest
* PDF: probability distribution function
* LUT: look-up table
* LKHD: likelihood
