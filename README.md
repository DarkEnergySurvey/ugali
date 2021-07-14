[![Build](https://github.com/DarkEnergySurvey/ugali/actions/workflows/python-package.yml/badge.svg)](https://github.com/DarkEnergySurvey/ugali/actions/workflows/python-package.yml)
[![PyPI](https://img.shields.io/pypi/v/ugali.svg)](https://pypi.python.org/pypi/ugali)
[![Release](https://img.shields.io/github/release/DarkEnergySurvey/ugali.svg)](../../releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../../)
[![Github Downloads](https://img.shields.io/github/downloads/DarkEnergySurvey/ugali/total.svg)](../../releases)

Overview
--------

The ultra-faint galaxy likelihood (`ugali`) toolkit provides a set of python modules developed for maximum-likelihood-based studies of resolved satellite galaxies. The primary inputs are stellar object catalogs derived from optical photometric surveys and the coverage masks of those surveys. In addition, `ugali` ships with a set of synthetic isochrone libraries and catalogs of known resolved stellar systems.

Authors: [Keith Bechtol](https://github.com/bechtol) & [Alex Drlica-Wagner](https://github.com/kadrlica)

Installation
------------

There are several ways to install `ugali`.

The most robust way is to follow the installation procedure for the automated GitHub Actions build documented in [python-package.yml](.github/workflows/python-package.yml). This installation creates a `conda` environment with the necessary dependencies and installs `ugali`.
```bash
# Create and activate conda environment
conda create -n ugali-env numpy scipy matplotlib astropy healpy pyyaml emcee nose fitsio corner -c conda-forge -c kadrlica
source activate ugali-env

# Clone source code from the parent repository
git clone https://github.com/DarkEnergySurvey/ugali.git && cd ugali

# Install just the python source code
python setup.py install 

# Install source code with a minimal set of isochrone and catalog libraries
python setup.py install --isochrones --catalogs
```

In theory, the easiest way to get a stable release of `ugali` is through [PyPi](https://pypi.python.org/pypi) using `pip`:
```bash
# Install just the source code
pip install ugali

# Install source code with a minimal set of isochrone and catalog libraries
pip install ugali --install-option "--isochrones" --install-option "--catalogs"
```

By default, the minimal isochrone and catalog libraries are installed into the directory specified by the `$UGALIDIR` environment variable (default: `$HOME/.ugali`). The download and unpacking of the isochrone and catalog files might make it appear that your `pip` installation has stalled. Unfortunately, `pip` [may not display a progress bar](https://github.com/pypa/pip/issues/2732#issuecomment-97119093) during this delay.

Auxiliary Libraries
-------------------

The `ugali` source code is distributed with several auxiliary libraries for isochrone generation and catalog matching. These libraries can be downloaded directly from the [releases](../../releases) page, and unpacked in your `$UGALIDIR`. For example, to install the [Bressan et al. 2012](http://adsabs.harvard.edu/abs/2012MNRAS.427..127B) isochrones for the DES survey:

```
cd $UGALIDIR
wget https://github.com/kadrlica/ugali/releases/download/v1.7.0rc0/ugali-des-bressan2012.tar.gz
tar -xzf ugali-des-bressan2012.tar.gz
```

The `UGALIDIR` environment variable is used to point to the isochrone and catalog libraries. If you install the isochrones in a non-standard location be sure to set `UGALIDIR` so `ugali` can find them:

```
export UGALIDIR=<PATH>
```

An experimental interface for downloading the isochrone and catalog libraries also exists through `setup.py`:
```
# To install the Bressan+ 2012 isochrones for the DES survey
python setup.py isochrones --survey des --model bressan2012

# If you have isochrones already installed, you may need to force
python setup.py isochrones --survey des --model bressan2012 --force

# To install all available DES isochrones
python setup.py isochrones --survey des

# To install all available Bressan+ 2012 isochrones
python setup.py isochrones --model bressan2012

# To install the catalog libraries
python setup.py catalogs
```

Usage Examples
--------------

Several examples can be found as jupyter notebooks in the [notebooks](notebooks/) directory.

Additional Information
----------------------

#### Dependencies
These should mostly be taken care of by PyPi with a `pip install`.
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](http://matplotlib.org/)
* [astropy](http://www.astropy.org/)
* [healpy](https://github.com/healpy/healpy)
* [emcee](http://dan.iel.fm/emcee/current/)
* [pyyaml](http://pyyaml.org/)
* [fitsio](https://github.com/esheldon/fitsio)

#### Isochrones
The isochrones used by `ugali` come dominantly from:
* PARSEC isochrones from the Padova group (http://stev.oapd.inaf.it/cgi-bin/cmd)
* Dartmouth isochrones (http://stellar.dartmouth.edu/models/isolf_new.html)
* MESA isochrones (http://waps.cfa.harvard.edu/MIST/interp_isos.html)

More information can be found in the [isochrone](ugali/isochrone) module.

#### Abbreviations
* CMD: color-magnitude diagram
* ROI: region of interest
* PDF: probability distribution function
* LKHD: likelihood
* IMF: initial mass function
* LUT: look-up table
