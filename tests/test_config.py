#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import os
import numpy as np
import ugali.utils.config

CONFIG='ugali/config/config_test.yaml'

def test_config():
    config = ugali.utils.config.Config(CONFIG)
    np.testing.assert_equal(CONFIG, config.filename)
    np.testing.assert_equal(len(config.filenames),768)

    np.testing.assert_equal(config.filenames['pix'].compressed()[0],687)
    np.testing.assert_equal(config.filenames['catalog'].compressed()[0],
                            './healpix/catalog_hpx0687.fits')
    np.testing.assert_equal(config.filenames['mask_1'].compressed()[0],
                            './mask/maglim_g_hpx0687.fits')
    np.testing.assert_equal(config.filenames['mask_2'].compressed()[0],
                            './mask/maglim_r_hpx0687.fits')

    np.testing.assert_equal(config.likefile,'./scan/scan_%08i_%s.fits')
    np.testing.assert_equal(config.mergefile,'./scan/merged_scan.fits')
    np.testing.assert_equal(config.roifile,'./scan/merged_roi.fits')

    np.testing.assert_equal(config.labelfile,'./search/merged_labels.fits')
    np.testing.assert_equal(config.objectfile,'./search/ugali_objects.fits')
    np.testing.assert_equal(config.assocfile,'./search/ugali_assocs.fits')
    np.testing.assert_equal(config.candfile,'./search/ugali_candidates.fits')

    np.testing.assert_equal(config.mcmcfile,'./mcmc/%s_mcmc.npy')

    
if __name__ == '__main__':
    test_config()
