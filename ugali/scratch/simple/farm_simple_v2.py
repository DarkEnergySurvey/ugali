import os
import time
import subprocess
import glob
import pyfits
import healpy
import numpy

import ugali.utils.healpix

############################################################

username = 'bechtol'

#datadir = '/project/kicp/bechtol/des/mw_substructure/y2n/data/catalog/hpx/cat'
datadir = '/project/kicp/bechtol/des/mw_substructure/y2n/data/catalog/v6/hpx' # v6
infiles = glob.glob('%s/cat_hpx_*.fits'%(datadir))

print 'Pixelizing...'
pix_16 = [] # Equatorial coordinates, RING ordering scheme
for infile in infiles:
    pix_16.append(int(infile.split('.fits')[0].split('_')[-1]))

#import pylab
#pylab.ion()
#npix_16 = healpy.nside2npix(16)
#m = numpy.tile(healpy.UNSEEN, npix_16)
#for pix in pix_16:
#    m[pix] = 1.
#healpy.mollview(m, xsize=1600)
#import sys
#sys.exit()

############################################################

for ii in range(0, len(pix_16)):
    ra, dec = ugali.utils.healpix.pixToAng(16, pix_16[ii])
    #ra, dec = 359.04, -59.63 # JUST A TEST, Tuc III (?)
    #ra, dec = 56.36, -60.44 # JUST A TEST, New
    #ra, dec = 94., -37. # JUST A TEST, point near edge of survey
    #ra, dec = 43.89, -54.12 # JUST A TEST, Hor I

    #if pix_16[ii] not in [2703, 2767]:
    #    continue

    print '(%i/%i)'%(ii, len(pix_16))

    while True:
        n_output = subprocess.Popen('ls slurm*.out | wc', shell=True, 
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0]
        if n_output.isdigit():
            os.system('rm slurm*.out')
        n_submitted = int(subprocess.Popen('squeue -u %s | wc\n'%username, shell=True, 
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate()[0].split()[0]) - 1
        if n_submitted < 100:
            break
        else:
            print '%i jobs already in queue, waiting ...'%(n_submitted)
            time.sleep(15)

    jobname = 'simple_binner'
    #batch = 'sbatch --account=kicp --partition=kicp-ht --job-name=%(jobname)s --mem=10000 '
    batch = 'sbatch --account=kicp --partition=kicp-ht --job-name=%s --mem=10000 '%(jobname)
    command = 'simple_v7.py %.2f %.2f'%(ra, dec)
    command_queue = batch + command
    print command_queue
    #os.system('./' + command) # Run locally
    os.system(command_queue) # Submit to queue

    #break
