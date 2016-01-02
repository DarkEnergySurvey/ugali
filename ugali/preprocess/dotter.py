#!/usr/bin/env python
"""
Script to automate the process of generating an isochrone library using the Dotter isochrones.

Download isochrones from
http://stellar.dartmouth.edu/models/isolf_new.html
"""

import urllib2
import os
import sys
import numpy

import ugali.utils.logger
import ugali.utils.shell

############################################################

def getDotterIsochrone(age, z, outdir, afe='+0.4', hel='Y=0.245+1.5*Z', clr='DECam', clobber=False):
    """
    See Vargas et al. 2013 for the distribution of alpha elements in
    dSphs: http://adsabs.harvard.edu/abs/2013ApJ...767..134V
    
    Josh Simon remarks: For stars at [Fe/H] > -2, [a/Fe] tends to be
    around zero. [Note, though, that this paper does not attempt to do
    any membership classification, it just accepts the lists from
    Simon & Geha 2007.  I suspect now that we were not sufficiently
    conservative on selecting members in those early days, and so some
    of the relatively metal-rich stars may in fact be foreground Milky
    Way stars.]  More metal-poor stars tend to average more like
    [a/Fe] = 0.4-0.5.  Fig. 5 of Frebel et al. (2014) shows similar
    plots for individual elements from high-resolution spectra.  Given
    these data, plus the empirical fact that the mean metallicities of
    the ultra-faint dwarfs are almost universally [Fe/H] < -2, I guess
    I would say [a/Fe] = 0.3 is probably the best compromise.
    """

    # KCB: This check won't work if the isochrone files have to be nenamed
    outfile = '%s/iso_a%.1f_z%.5f_temp.dat'%(outdir, age, z)
    if os.path.exists(outfile) and not clobber:
        ugali.utils.logger.logger.warning('Found %s; skipping...'%(outfile))
        return

    ugali.utils.logger.logger.info('Downloading isochrone: %s (age=%.2fGyr, metallicity=%g)'%(os.path.basename(outfile), age, z))

    ##### Hard-coded options from the Dotter web interface -- do not use blindly #####

    dict_clr = {'SDSS ugriz': '11',
                'PanSTARRS': '12',
                'DECam': '14'}
    if clr not in dict_clr.keys():
        ugali.utils.logger.logger.error('Photometric system %s not in available options: %s'%(clr, ', '.join(dict_clr.keys())))
        sys.exit()

    dict_afe = {'-0.2': '1',
                '0 (scaled-solar)': '2',
                '+0.2': '3',
                '+0.4': '4',
                '+0.6': '5',
                '+0.8': '6'}
    if afe not in dict_afe.keys():
        ugali.utils.logger.logger.error('[alpha/Fe] ratio %s not in available options: %s'%(afe, ', '.join(dict_afe.keys())))
        sys.exit()

    dict_hel = {'Y=0.245+1.5*Z': '1',
                'Y=0.33': '2',
                'Y=0.40': '3'}
    if hel not in dict_hel.keys():
        ugali.utils.logger.logger.error('Helium abundance %s not in available options: %s'%(hel, ', '.join(dict_hel.keys())))
        sys.exit()

    #####

    z_solar = 0.02 # Approximately the Dotter convention
    #z_solar = 0.0163 # Dotter convention
    feh = numpy.log10(z / z_solar)
    #print feh
    query = 'http://stellar.dartmouth.edu/models/isolf_new.php?int=1&out=1&age=%.1f&feh=%.5f&hel=%i&afe=%i&clr=%i&flt=&bin=&imf=1&pls=&lnm=&lns='%(age, 
                                                                                                                                                   feh, 
                                                                                                                                                   int(dict_hel[hel]), 
                                                                                                                                                   int(dict_afe[afe]), 
                                                                                                                                                   int(dict_clr[clr]))
    response = urllib2.urlopen(query)
    page_source = response.read()
    isochrone_id = page_source.split('tmp/tmp')[-1].split('.iso')[0]
    infile = 'http://stellar.dartmouth.edu/models/tmp/tmp%s.iso'%(isochrone_id)
    command = 'wget -q %s -O %s'%(infile, outfile)
    os.system(command)

    # Now rename the output file based on Z effective
    reader = open(outfile)
    lines = reader.readlines()
    reader.close()
    z_eff = float(lines[3].split()[4])

    outfile_new = '%s/iso_a%.1f_z%.5f.dat'%(outdir, age, z_eff)
    #outfile = 'iso_a%.1f_z%.5f.dat'%(age, z)
    #command = 'mv tmp%s.iso %s'%(isochrone_id, outfile_new)
    command = 'mv %s %s'%(outfile, outfile_new)

    #print command
    os.system(command)

############################################################

if __name__ == "__main__":
    #import ugali.utils.parser
    #description = "Download Dotter isochrones"
    #parser = ugali.utils.parser.Parser(description=description)

    #from ugali.utils.config import Config
    #config = Config(opts.config)

    #outdir = '/Users/keithbechtol/Documents/DES/projects/mw_substructure/des/isochrones/dotter_v3/'
    outdir = '/Users/keithbechtol/Documents/DES/projects/mw_substructure/des/isochrones/dotter_v4/'

    if outdir is None: outdir = './'
    ugali.utils.shell.mkdir(outdir)

    # Range used for PARSEC isochrone library
    #z_array = 10**numpy.linspace(numpy.log10(1.e-4), numpy.log10(1.e-2), 33)
    #age_array = numpy.arange(1., 13.5 + 0.1, 0.1)
    # Going to higher metallicity, [Fe/H] = -2.49
    z_array = 10**numpy.linspace(numpy.log10(6.5e-5), numpy.log10(5.e-3), 33)
    age_array = numpy.arange(6., 13.5 + 0.1, 0.1)
    
    # Test
    #getDotterIsochrone(12.2, 1.1e-4, outdir=outdir)
    #print '\nIf this were not a test, this script would download the following isochrone grid...'

    print '\nAges:'
    print age_array
    print '\nMetallicities:'
    print z_array

    count = 0
    for z in z_array:
        for age in age_array:
            print '(%i/%i)'%(count, len(age_array) * len(z_array))
            getDotterIsochrone(age, z, outdir=outdir)
            count += 1

############################################################
