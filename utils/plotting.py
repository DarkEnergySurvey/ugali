"""
Basic plotting tools.
"""

import numpy
import pylab
import healpy

pylab.ion()

############################################################

def histogram(title, title_x, title_y,
              x, bins_x):
    """
    Plot a basic histogram.
    """
    pylab.figure()
    pylab.hist(x, bins_x)
    pylab.xlabel(title_x)
    pylab.ylabel(title_y)
    pylab.title(title)

############################################################

def twoDimensionalHistogram(title, title_x, title_y,
                            z, bins_x, bins_y,
                            lim_x=None, lim_y=None,
                            vmin=None, vmax=None):
    """
    Create a two-dimension histogram plot or binned map.

    If using the outputs of numpy.histogram2d, remember to transpose the histogram.

    INPUTS
    """
    pylab.figure()

    mesh_x, mesh_y = numpy.meshgrid(bins_x, bins_y)

    if vmin != None and vmin == vmax:
        pylab.pcolor(mesh_x, mesh_y, z)
    else:
        pylab.pcolor(mesh_x, mesh_y, z, vmin=vmin, vmax=vmax)
    pylab.xlabel(title_x)
    pylab.ylabel(title_y)
    pylab.title(title)
    pylab.colorbar()

    if lim_x:
        pylab.xlim(lim_x[0], lim_x[1])
    if lim_y:
        pylab.ylim(lim_y[0], lim_y[1])
        
############################################################

def twoDimensionalScatter(title, title_x, title_y,
                          x, y,
                          lim_x = None, lim_y = None,
                          color = 'b', size = 20, alpha=None):
    """
    Create a two-dimensional scatter plot.

    INPUTS
    """
    pylab.figure()

    pylab.scatter(x, y, c=color, s=size, alpha=alpha, edgecolors='none')
    
    pylab.xlabel(title_x)
    pylab.ylabel(title_y)
    pylab.title(title)
    if type(color) is not str:
        pylab.colorbar()

    if lim_x:
        pylab.xlim(lim_x[0], lim_x[1])
    if lim_y:
        pylab.ylim(lim_y[0], lim_y[1])

############################################################

def zoomedHealpixMap(title, map, lon, lat, radius,
                     xsize=1000, **kwargs):
    """
    Inputs: lon (deg), lat (deg), radius (deg)
    """
    reso = 60. * 2. * radius / xsize # Deg to arcmin
    healpy.gnomview(map=map, rot=[lon, lat, 0], title=title, xsize=xsize, reso=reso, degree=False, **kwargs)
    
############################################################
