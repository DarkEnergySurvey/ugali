"""
Tools for binning data.
"""

import numpy
import numpy as np
import collections

############################################################

def centers(array):
    """
    Centers corresponding to bin edges (assuming regular linear intervals).
    """
    return array[0: -1] + (0.5 * (array[1] - array[0]))

############################################################

def take2D(histogram, x, y, bins_x, bins_y):
    """
    Take the value from a two-dimensional histogram from the bin corresponding to (x, y).
    """
    histogram = numpy.array(histogram)
    
    if numpy.isscalar(x):
        x = [x]
    if numpy.isscalar(y):
        y = [y]

    bins_x[-1] += 1.e-10 * (bins_x[-1] - bins_x[-2]) # Numerical stability
    bins_y[-1] += 1.e-10 * (bins_y[-1] - bins_y[-2])

    #return numpy.take(histogram, (histogram.shape[1] * (numpy.digitize(y, bins_y) - 1)) + (numpy.digitize(x, bins_x) - 1))

    # Return numpy.nan for entries which are outside the binning range on either axis
    index = (histogram.shape[1] * (numpy.digitize(y, bins_y) - 1)) + (numpy.digitize(x, bins_x) - 1)
    index_clipped = numpy.clip(index, 0, (histogram.shape[0] * histogram.shape[1]) - 1)
    val = numpy.take(histogram, index_clipped)

    outlier_x = numpy.logical_or(x < bins_x[0], x > bins_x[-1])
    outlier_y = numpy.logical_or(y < bins_y[0], y > bins_y[-1])
    outlier = numpy.logical_or(outlier_x, outlier_y)
    val[outlier] = numpy.nan

    return val 
    
############################################################

def cloudInCells(x, y, bins, weights=None):
    """
    Use cloud-in-cells binning algorithm. Only valid for equal-spaced linear bins.
    http://ta.twi.tudelft.nl/dv/users/Lemmens/MThesis.TTH/chapter4.html#tth_sEc2
    http://www.gnu.org/software/archimedes/manual/html/node29.html

    INPUTS:
        x: array of x-values
        y: array or y-values
        bins: [bins_x, bins_y] format, where bins_x corresponds to the bin edges along x-axis
        weights[None]: optionally assign a weight to each entry
    OUTPUTS:
        histogram:
        bins_x:
        bins_y:
    """

    # For consistency, the variable names should be changed in this function, but low priority...
    
    x_bins = numpy.array(bins[0])
    delta_x = x_bins[1] - x_bins[0]
    # Overflow and underflow bins
    x_bins = numpy.insert(x_bins, 0, x_bins[0] - delta_x)
    x_bins = numpy.append(x_bins, x_bins[-1] + delta_x)
    y_bins = numpy.array(bins[1])
    delta_y = y_bins[1] - y_bins[0]
    y_bins = numpy.insert(y_bins, 0, y_bins[0] - delta_y)
    y_bins = numpy.append(y_bins, y_bins[-1] + delta_y)

    x_bound_cut = numpy.logical_and(x >= x_bins[0], x <= x_bins[-1])
    y_bound_cut = numpy.logical_and(y >= y_bins[0], y <= y_bins[-1])
    bound_cut = numpy.logical_and(x_bound_cut, y_bound_cut)

    if not numpy.any(weights):
        bound_weights = numpy.ones(len(x))[bound_cut]
    else:
        bound_weights = numpy.array(weights)[bound_cut]

    x_vals = numpy.array(x)[bound_cut]
    y_vals = numpy.array(y)[bound_cut]

    x_width = x_bins[1] - x_bins[0]
    y_width = y_bins[1] - y_bins[0]

    x_centers = x_bins[0: -1] + (0.5 * x_width)  
    y_centers = y_bins[0: -1] + (0.5 * y_width)

    dx = x_vals - x_centers[numpy.digitize(x_vals, x_bins) - 1]
    dy = y_vals - y_centers[numpy.digitize(y_vals, y_bins) - 1]

    ux = ((dx / x_width) * (dx >= 0)) +\
         ((1. + (dx / x_width)) * (dx < 0))
    lx = 1. - ux

    uy = ((dy / y_width) * (dy >= 0)) +\
         ((1. + (dy / y_width)) * (dy < 0))
    ly = 1. - uy

    new_x_vals = []
    new_y_vals = []
    cell_weights = []

    # 4 corners
    new_x_vals.append(x_vals + (0.5 * x_width))
    new_y_vals.append(y_vals + (0.5 * y_width))
    cell_weights.append(bound_weights * ux * uy)

    new_x_vals.append(x_vals + (0.5 * x_width))
    new_y_vals.append(y_vals - (0.5 * y_width))
    cell_weights.append(bound_weights * ux * ly)
    
    new_x_vals.append(x_vals - (0.5 * x_width))
    new_y_vals.append(y_vals + (0.5 * y_width))
    cell_weights.append(bound_weights * lx * uy)
    
    new_x_vals.append(x_vals - (0.5 * x_width))
    new_y_vals.append(y_vals - (0.5 * y_width))
    cell_weights.append(bound_weights * lx * ly)

    new_x_vals = numpy.concatenate(new_x_vals)
    new_y_vals = numpy.concatenate(new_y_vals)
    cell_weights = numpy.concatenate(cell_weights)

    result = numpy.histogram2d(new_x_vals, new_y_vals,
                               bins = [x_bins, y_bins],
                               weights = cell_weights)[0]
 
    result = numpy.transpose(result[1: result.shape[0] - 1])[1: result.shape[1] - 1]

    return result, x_bins, y_bins


def kernelDensity(x, y, bins, weights=None, fx=1, fy=1):
    nx,ny = (len(bins[0])-1)*fx,(len(bins[1])-1)*fy
    xmin,xmax = bins[0].min(),bins[0].max()
    ymin,ymax = bins[1].min(),bins[1].max()
    # Correct weights for CMD bin size
    if weights is not None:
        weights = np.asarray(weights)
        weights *= 1/float(fx*fy)
    grid = fast_kde(x,y,gridsize=(nx,ny),extents=[xmin,xmax,ymin,ymax])
    results = sum_chunks(grid,fx,fy)
    return results,bins[0],bins[1]
    
############################################################

"""
A faster gaussian kernel density estimate (KDE).
Intended for computing the KDE on a regular grid (different use case than 
scipy's original scipy.stats.kde.gaussian_kde()).
@author: Joe Kington

@note: http://pastebin.com/LNdYCZgw
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.signal

def fast_kde(x, y, gridsize=(200,200), extents=None, nocorrelation=False, weights=None):
    """
    Performs a gaussian kernel density estimate over a regular grid using a
    convolution of the gaussian kernel with a 2D histogram of the data.

    This function is typically several orders of magnitude faster than 
    scipy.stats.kde.gaussian_kde for large (>1e7) numbers of points and 
    produces an essentially identical result.

    Input:
        x: The x-coords of the input data points
        y: The y-coords of the input data points
        gridsize: (default: 200x200) A (nx,ny) tuple of the size of the output 
            grid
        extents: (default: extent of input data) A (xmin, xmax, ymin, ymax)
            tuple of the extents of output grid
        nocorrelation: (default: False) If True, the correlation between the
            x and y coords will be ignored when preforming the KDE.
        weights: (default: None) An array of the same shape as x & y that 
            weighs each sample (x_i, y_i) by each value in weights (w_i).
            Defaults to an array of ones the same size as x & y.
    Output:
        A gridded 2D kernel density estimate of the input points. 
    """
    #---- Setup --------------------------------------------------------------
    x, y = np.asarray(x), np.asarray(y)
    x, y = np.squeeze(x), np.squeeze(y)
    
    if x.size != y.size:
        raise ValueError('Input x & y arrays must be the same size!')

    nx, ny = gridsize
    n = x.size

    if weights is None:
        # Default: Weight all points equally
        weights = np.ones(n)
    else:
        weights = np.squeeze(np.asarray(weights))
        if weights.size != x.size:
            raise ValueError('Input weights must be an array of the same size'
                    ' as input x & y arrays!')

    # Default extents are the extent of the data
    if extents is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = map(float, extents)
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    #---- Preliminary Calculations -------------------------------------------

    # First convert x & y over to pixel coordinates
    # (Avoiding np.digitize due to excessive memory usage in numpy < v1.5!)
    # http://stackoverflow.com/q/8805601/
    xyi = np.vstack((x,y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T

    # Next, make a 2D histogram of x & y
    # Avoiding np.histogram2d due to excessive memory usage with many points
    # http://stackoverflow.com/q/8805601/
    grid = sp.sparse.coo_matrix((weights, xyi), shape=(nx, ny)).toarray()

    # Calculate the covariance matrix (in pixel coords)
    cov = np.cov(xyi)

    if nocorrelation:
        cov[1,0] = 0
        cov[0,1] = 0

    # Scaling factor for bandwidth
    scotts_factor = np.power(n, -1.0 / 6) # For 2D

    #---- Make the gaussian kernel -------------------------------------------

    # First, determine how big the kernel needs to be
    std_devs = np.diag(np.sqrt(cov))
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)

    # Determine the bandwidth to use for the gaussian kernel
    inv_cov = np.linalg.inv(cov * scotts_factor**2) 

    # x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    xx = np.arange(kern_nx, dtype=np.float) - kern_nx / 2.0
    yy = np.arange(kern_ny, dtype=np.float) - kern_ny / 2.0
    xx, yy = np.meshgrid(xx, yy)

    # Then evaluate the gaussian function on the kernel grid
    kernel = np.vstack((xx.flatten(), yy.flatten()))
    kernel = np.dot(inv_cov, kernel) * kernel 
    kernel = np.sum(kernel, axis=0) / 2.0 
    kernel = np.exp(-kernel) 
    kernel = kernel.reshape((kern_ny, kern_nx))

    #---- Produce the kernel density estimate --------------------------------

    # Convolve the gaussian kernel with the 2D histogram, producing a gaussian
    # kernel density estimate on a regular grid
    grid = sp.signal.convolve2d(grid, kernel, mode='same', boundary='fill').T

    ### ADW: Commented out for 
    ### # Normalization factor to divide result by so that units are in the same
    ### # units as scipy.stats.kde.gaussian_kde's output.  
    ### norm_factor = 2 * np.pi * cov * scotts_factor**2
    ### norm_factor = np.linalg.det(norm_factor)
    ### norm_factor = n * dx * dy * np.sqrt(norm_factor)
    ###  
    ### # Normalize the result
    ### grid /= norm_factor

    return grid

def sum_chunks(data,fx=4,fy=4):
    y,x = data.shape
    binned = data.reshape(y//fy, fy, x//fx, fx)
    return binned.sum(axis=3).sum(axis=1)

############################################################

def reverseHistogram(data,bins=None):
    """             
    Bins data using numpy.histogram and calculates the
    reverse indices for the entries like IDL.
    Parameters:
    data  : data to pass to numpy.histogram
    bins  : bins to pass to numpy.histogram 
    Returns: 
    hist  : bin content output by numpy.histogram 
    edges : edges output from numpy.histogram 
    rev   : reverse indices of entries in each bin 
    Using Reverse Indices: 
        h,e,rev = histogram(data, bins=bins) 
        for i in range(h.size):  
            if rev[i] != rev[i+1]: 
                # data points were found in this bin, get their indices
                indices = rev[ rev[i]:rev[i+1] ] 
                # do calculations with data[indices] ...  
    """
    if bins is None: bins = numpy.arange(data.max()+2)
    hist, edges = numpy.histogram(data, bins=bins)
    digi = numpy.digitize(data.flat,bins=numpy.unique(data)).argsort()
    rev = numpy.hstack( (len(edges), len(edges) + numpy.cumsum(hist), digi) )
    return hist,edges,rev
    
def binnedMedian(x,y,xbins=None):
    hist,edges,rev = reverseHistogram(x,bins=xbins)
    avg_x = (edges[:-1]+edges[1:])/2.
    med_y = []
    for i,n in enumerate(hist):
        indices = rev[ rev[i]:rev[i+1] ] 
        med_y.append( numpy.median(y[indices]) )
    return avg_x, numpy.array(med_y)
