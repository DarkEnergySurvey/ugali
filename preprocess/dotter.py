"""
Script to automate the process of generating an isochrone library using the Dotter isochrones.
"""

import urllib2
import os
import sys
import numpy

############################################################

def getDotterIsochrone(age, z, afe='+0.4', clr='DECam'):
    """
    See Vargas et al. 2013 for the distribution of alpha elements in dSphs
    http://adsabs.harvard.edu/abs/2013ApJ...767..134V
    
    Josh Simon remarks:
    For stars at [Fe/H] > -2, [a/Fe] tends to be around zero. [Note, though, that this paper does not attempt to 
    do any membership classification, it just accepts the lists from Simon & Geha 2007.  I suspect now that we 
    were not sufficiently conservative on selecting members in those early days, and so some of the relatively 
    metal-rich stars may in fact be foreground Milky Way stars.]  More metal-poor stars tend to average more 
    like [a/Fe] = 0.4-0.5.  Fig. 5 of Frebel et al. (2014) shows similar plots for individual elements from 
    high-resolution spectra.  Given these data, plus the empirical fact that the mean metallicities of the 
    ultra-faint dwarfs are almost universally [Fe/H] < -2, I guess I would say [a/Fe] = 0.3 is probably the 
    best compromise.
    """

    dict_clr = {'SDSS ugriz': '11',
                'PanSTARRS': '12',
                'DECam': '14'}
    if clr not in dict_clr.keys():
        sys.exit('ERROR: Photometric system %s not in available options: %s'%(clr, ', '.join(dict_clr.keys())))

    dict_afe = {'-0.2': '0',
                '0 (scaled-solar)': '1',
                '+0.2': '2',
                '+0.4': '3',
                '+0.6': '4',
                '+0.8': '5'}
    if afe not in dict_afe.keys():
        sys.exit('ERROR: [alpha/Fe] ratio %s not in available options: %s'%(afe, ', '.join(dict_afe.keys())))

    z_solar = 0.02 # Dotter convention
    feh = numpy.log10(z / z_solar)
    query = 'http://stellar.dartmouth.edu/models/isolf_new.php?int=1&out=1&age=%.1f&feh=%.5f&hel=1&afe=%i&clr=%i&flt=&bin=&imf=1&pls=&lnm=&lns='%(age, feh, dict_afe[afe], dict_clr[clr])
    response = urllib2.urlopen(query)
    page_source = response.read()
    isochrone_id = page_source.split('tmp/tmp')[-1].split('.iso')[0]
    infile = 'http://stellar.dartmouth.edu/models/tmp/tmp%s.iso'%(isochrone_id)
    command = 'wget -q %s'%(infile)
    os.system(command)
    outfile = 'iso_a%.1f_z%.5f.dat'%(age, z)
    command = 'mv tmp%s.iso %s'%(isochrone_id, outfile)
    print command
    os.system(command)

############################################################

if __name__ == "__main__":

    # Test
    getDotterIsochrone(12.2, 1.1e-4)

    z_array = 10**numpy.linspace(numpy.log10(1.e-4), numpy.log10(1.e-2), 33)
    age_array = numpy.arange(1., 13.5 + 0.1, 0.1)

    #for z in z_array:
    #    for age in age_array:
    #        getDotterIsochrone(age, z)

############################################################
