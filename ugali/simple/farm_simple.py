#!/usr/bin/env python
"""
Perform simple binning search
"""
__author__ = "Sidney Mau"

import os
import time
import subprocess
import glob
import astropy.io.fits as pyfits
import healpy
import numpy

import ugali.utils.healpix

import yaml

############################################################

with open('config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

nside = cfg['nside']
datadir = cfg['datadir']

results_dir = os.path.join(os.getcwd(), cfg['results_dir'])
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

log_dir = os.path.join(os.getcwd(), cfg['results_dir'], cfg['log_dir'])
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

#infiles = glob.glob('%s/cat_hpx_*.fits'%(datadir))
infiles = glob.glob ('%s/y3a2_ngmix_cm_*.fits'%(datadir))
############################################################

print('Pixelizing...')
pix_nside = [] # Equatorial coordinates, RING ordering scheme
for infile in infiles:
    pix_nside.append(int(infile.split('.fits')[0].split('_')[-1]))

############################################################

for ii in range(0, len(pix_nside)):
    ra, dec = ugali.utils.healpix.pixToAng(nside, pix_nside[ii])

    print('({}/{})').format(ii, len(pix_nside))

    #pix_nside[ii] = pix_nside_select
    logfile = '%s/results_nside_%s_%i.log'%(log_dir, nside, pix_nside[ii])
    batch = 'csub -n 20 -o %s '%logfile # q local for debugging
    command = 'python search_algorithm.py %.2f %.2f'%(ra, dec)
    command_queue = batch + command
    print(command_queue)
    #os.system('./' + command) # Run locally
    os.system(command_queue) # Submit to queue
