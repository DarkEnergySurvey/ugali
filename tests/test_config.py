#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import os
import ugali.utils.config

"""
catalog:
  #infile: None
  dirname: /u/ki/kadrlica/des/data/y3a2/gold/1.2/healpix
  basename: "catalog_hpx%04i.fits"
  lon_field: RA
  lat_field: DEC
  # Color always defined as mag_1 - mag_2
  objid_field       : COADD_OBJECT_ID
  mag_1_band        : &band1 g
  mag_1_field       : PSF_MAG_SFD_G
  mag_err_1_field   : PSF_MAG_ERR_G
  mag_2_band        : &band2 r
  mag_2_field       : PSF_MAG_SFD_R
  mag_err_2_field   : PSF_MAG_ERR_R
  # True = band 1 is detection band; False = band 2 is detection band     
  band_1_detection  : &detection True
  mc_source_id_field: MC_SOURCE_ID
  selection         : "((self.data['FLAG_FOREGROUND'] & 16) == 0)"
  #selection         : null

coords:
  nside_catalog   : 8      # Size of patches for catalog binning
  nside_mask      : 64     # Size of patches for mask creation
  nside_likelihood: 256    # Size of target pixel
  nside_pixel     : 4096   # Size of pixels within target region
  roi_radius      : 2.0    # Outer radius of background annulus
  roi_radius_annulus: 0.5  # Inner radius of background annulus
  roi_radius_interior: 0.5 # Radius of interior region for likelihood analysis
  coordsys : cel           # Coordinate system ['CEL','GAL']
  proj_type: ait

mask:
  dirname    : /u/ki/kadrlica/des/data/y3a2/gold/1.2/split
  basename_1 : "maglim_g_hpx%04i.fits"
  basename_2 : "maglim_r_hpx%04i.fits"
  minimum_solid_angle: 0.1 # deg^2

likelihood:
  delta_mag: 0.01 # 1.e-3 

output:
  likedir    : ./scan
  searchdir  : ./search
  mcmcdir    : ./mcmc_v01
  simdir     : ./sims
  resultdir  : ./results
  plotdir    : ./plots
  likefile   : "scan_%08i_%s.fits"
  mergefile  :  merged_scan.fits
  roifile    :  merged_roi.fits
  labelfile  :  merged_labels.fits
  objectfile :  ugali_objects.fits
  assocfile  :  ugali_assocs.fits
  candfile   :  ugali_candidates.fits
  mcmcfile   : "%s_mcmc.npy"
  simfile    : "sims_%04i.fits"

batch:
  max_jobs: 200
  chunk: 25
"""

def test_load_config():
    dirname = os.path.dirname(os.path.realpath(ugali.utils.config.__file__))
    configfile = os.path.join(dirname,'..','config/config_y3a2_cel.yaml')
    config = ugali.utils.config.Config(configfile)
    
    assert configfile == config.filename
    
if __name__ == '__main__':
    test_load_config()
