#!/usr/bin/env python
"""
Create simple binner style plots for ugali or simple candidate lists
"""
__author__ = "Sidney Mau"

import os
import time
import subprocess
import glob
import pyfits
import healpy
import numpy
import numpy as np

import ugali.utils.healpix
import fitsio as fits

import yaml

############################################################

with open('config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

jobs = cfg['batch']['jobs']

candidate_list = cfg[cfg['data']]['candidate_list']

save_dir = os.path.join(os.getcwd(), cfg['output']['save_dir'])
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

log_dir = os.path.join(os.getcwd(), cfg['output']['save_dir'], cfg['output']['log_dir'])
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

if candidate_list.endswith('.csv'):
    candidate_list = np.genfromtxt(candidate_list, delimiter=',', names=['SIG', 'RA', 'DEC', 'MODULUS', 'r'])[1:] #, 'association', 'association_angsep'])[1:]
    candidate_list = candidate_list[candidate_list['SIG'] > 5.5] # get arg from standard input?
elif candidate_list.endswith('.fits'):
    candidate_list = fits.read(candidate_list)

############################################################

print('{} candidates found...').format(len(candidate_list))

############################################################

for candidate in range(len(candidate_list)):
    try: # simple
        sig = round(candidate_list[candidate]['SIG'], 2)
    except: # ugali
        sig = round(candidate_list[candidate]['TS'], 2)
    ra      = round(candidate_list[candidate]['RA'], 2)
    dec     = round(candidate_list[candidate]['DEC'], 2)
    mod     = round(candidate_list[candidate]['MODULUS'], 2)

    logfile = '{}/candidate_{}_{}.log'.format(log_dir, ra, dec)
    batch = 'csub -n {} -o {} '.format(jobs, logfile) # q local for debugging
    #command = 'python make_plot.py %.2f %.2f %.2f'%(ra, dec, mod, sig)
    command = 'python make_plot.py {} {} {} {}'.format(ra, dec, mod, sig)
    command_queue = batch + command
    print command_queue
    #os.system('./' + command) # Run locally
    os.system(command_queue) # Submit to queue
