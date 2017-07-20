import numpy
import scipy.interpolate

############################################################

def cutIsochronePath(g, r, g_err, r_err, isochrone, radius=0.1, return_all=False):
    """
    Cut to identify objects within isochrone cookie-cutter.
    """
    #print dir(isochrone)
    #print isochrone.age
    #print isochrone.metallicity
    #print isochrone.distance_modulus

    index_transition = numpy.nonzero(isochrone.stage > 3)[0][0] + 1    
    mag_1_rgb = isochrone.mag_1[0: index_transition] + isochrone.distance_modulus
    mag_2_rgb = isochrone.mag_2[0: index_transition] + isochrone.distance_modulus
    #print mag_1_rgb
    #print mag_2_rgb
    #print numpy.min(mag_1_rgb), numpy.max(mag_1_rgb)
    mag_1_rgb = mag_1_rgb[::-1]
    mag_2_rgb = mag_2_rgb[::-1]
    
    f_isochrone = scipy.interpolate.interp1d(mag_1_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = numpy.fabs((g - r) - f_isochrone(g))
    cut = (color_diff < numpy.sqrt(0.1**2 + r_err**2 + g_err**2))

    #print f_isochrone(17.)
    #print numpy.sum(cut)
    #pylab.figure('isochrone_cut')
    #pylab.scatter(g[0:1000], g[0:1000] - r[0:1000], edgecolor='none', s=1)
    #pylab.scatter(g[cut][0:1000], g[cut][0:1000] - r[cut][0:1000], edgecolor='none', s=1)
    #pylab.scatter(g[~cut][0:1000], g[~cut][0:1000] - r[~cut][0:1000], edgecolor='none', s=1)
    #pylab.plot(mag_1_rgb, mag_1_rgb - mag_2_rgb)
    #raw_input('hold up')

    mag_bins = numpy.arange(17, 24.1, 0.1)
    mag_centers = 0.5 * (mag_bins[1:] + mag_bins[0:-1])
    magerr = numpy.tile(0., len(mag_centers))
    for ii in range(0, len(mag_bins) - 1):
        cut_mag_bin = (g > mag_bins[ii]) & (g < mag_bins[ii + 1])
        magerr[ii] = numpy.median(numpy.sqrt(0.1**2 + r_err[cut_mag_bin]**2 + g_err[cut_mag_bin]**2))

    if return_all:
        return cut, mag_centers[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) + magerr)[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) - magerr)[f_isochrone(mag_centers) < 100]
    else:
        return cut

############################################################
