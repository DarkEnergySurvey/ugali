OVERVIEW
--------
Ultra-faint Galaxy Likelihood (UGaLi) is a set of python 
classes and functions used to search for ultra-faint satellite 
galaxies of the Milky Way using the stellar object samples 
derived from optical photometric surveys.

[Keith Bechtol](https://github.com/bechtol) & [Alex Drlica-Wagner](https://github.com/kadrlica)

USAGE EXAMPLES
--------------
Examples go here.

CODE REPOSITORY
---------------
* https://bitbucket.org/bechtol/ugali/
* https://github.com/kadrlica/ugali/

DEPENDENCIES
------------

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
Currently set up to use isochrones in the format produced by Leo Girardi's (Padova)
CMD tool, which has a web interface at http://stev.oapd.inaf.it/cgi-bin/cmd

CONVENTIONS
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
