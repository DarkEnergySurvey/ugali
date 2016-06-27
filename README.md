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

The easiest way is through PyPi using `pip`. To install just the `ugali` code base:
```
# To install just the source code
pip install ugali

# To also install the isochrone library
pip install ugali --install-option "--isochrones"
```

By default, the isochrone library is installed into `$HOME/.ugali`; however you can modify this on the command line with:
```
pip install ugali --install-option "--isochrones" --install-option "--isochrones-path <PATH-TO-ISOCHRONES>
```

To get the most up-to-date version of `ugali`, you can download the source code from github and install with `setup.py`:
```
git clone https://github.com/DarkEnergySurvey/ugali.git
cd ugali

# Install the python source code
python setup.py install

# To also install the isochrone library
python setup.py install --isochrones

# To specify the isochrone install path
python setup.py install --isochrones --isochrones-path=<PATH>
```

If you place the isochrones in a different directory be sure that `ugali` knows where to find them:
```
export UGALIDIR=$<INSTALL_PATH>/isochrones
```

Usage Examples
--------------
Examples go here.

Code Repository
---------------
* https://bitbucket.org/bechtol/ugali/
* https://github.com/kadrlica/ugali/

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
The ugali tools make use of a large library of stellar isochrones. These isochrones are derived from two different groups and are distributed as binary tarballs with releases of ugali.
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
