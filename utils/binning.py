"""
Tools for binning data.
"""

import numpy
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
    
############################################################

def reverseHistogram(data,bins=None):
    """                                                                                                    Bins data using numpy.histogram and calculates the
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
