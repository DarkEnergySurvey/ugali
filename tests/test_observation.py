#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import numpy as np
from ugali.utils.logger import logger
logger.setLevel(logger.WARN)

CONFIG='ugali/config/config_test.yaml'
LON = RA = 53.92
LAT = DEC = -54.05
POINTS = np.array([(LON,LAT), (53.81,-54.02), (54.17,-53.88), (52.59,-55.39)]).T
IDX = [0, 2537, 15032]

def test_roi():
    """ Testing ugali.observation.roi """
    import ugali.observation.roi
    roi = ugali.observation.roi.ROI(CONFIG, LON, LAT)

    # testing roi location
    np.testing.assert_equal(roi.lon, LON)
    np.testing.assert_equal(roi.lat, LAT)

    # testing the roi regions
    np.testing.assert_equal(len(roi.pixels),61327)
    np.testing.assert_equal(len(roi.pixels_interior),3828)
    np.testing.assert_equal(len(roi.pixels_annulus),57499)
    np.testing.assert_equal(len(roi.pixels_target),256)

    # test points
    np.testing.assert_equal(roi.inROI(POINTS[0],POINTS[1]),
                            np.array([True,True,True,True]))
    np.testing.assert_equal(roi.inAnnulus(POINTS[0],POINTS[1]),
                            np.array([False,False,False,True]))
    np.testing.assert_equal(roi.inInterior(POINTS[0],POINTS[1]),
                            np.array([True,True,True,False]))
    np.testing.assert_equal(roi.inTarget(POINTS[0],POINTS[1]),
                            np.array([True,True,False,False]))
    np.testing.assert_equal(roi.indexPixels(POINTS[0],POINTS[1],roi.pixels),
                            np.array([31034,30327,27527,54866]))

def test_catalog():
    """ Test ugali.observation.catalog """
    import ugali.observation.roi
    import ugali.observation.catalog

    filename='healpix/catalog_hpx0687.fits'
    catalog = ugali.observation.catalog.Catalog(CONFIG,filenames=filename)

    roi = ugali.observation.roi.ROI(CONFIG, LON, LAT)
    catalog2 = ugali.observation.catalog.Catalog(CONFIG,roi)

    # Testing global catalog properties
    np.testing.assert_equal(catalog,catalog2)
    np.testing.assert_equal(len(catalog),35600)

    cat = catalog.applyCut(IDX)
    np.testing.assert_equal(len(cat),3)
    cat2 = cat + cat
    np.testing.assert_equal(len(cat2),6)
    cat.write('tmp.fits')
    
    # Testing object properties
    np.testing.assert_allclose(cat.objid,[374645274,375702581,374652071])
    np.testing.assert_allclose(cat.mc_source_id,[0,0,0])

    np.testing.assert_allclose(cat.ra,[54.616013,53.839753,53.873778])
    np.testing.assert_allclose(cat.dec,[-53.898038, -54.095383, -53.966101])
    np.testing.assert_allclose(cat.glon,[265.821648,266.39369,266.199441])
    np.testing.assert_allclose(cat.glat,[-49.430314,-49.761495,-49.798647])
    np.testing.assert_allclose(cat.lon,cat.ra)
    np.testing.assert_allclose(cat.lat,cat.dec)

def test_mask():
    """ Test ugali.observation.mask """
    import ugali.observation.roi
    import ugali.observation.catalog
    import ugali.observation.mask

    roi = ugali.observation.roi.ROI(CONFIG, LON, LAT)
    catalog = ugali.observation.catalog.Catalog(CONFIG,roi=roi)
    mask = ugali.observation.mask.Mask(CONFIG, roi)

    # Test some operations
    np.testing.assert_allclose(mask.mask_roi_unique,[[0,0],[24,23.98]])
    np.testing.assert_equal(np.unique(mask.mask_roi_digi),[0,1])

    # Test the solid angle
    np.testing.assert_allclose(np.unique(mask.solid_angle_cmd),
                               [0., 0.15429397])
    np.testing.assert_equal(mask.mag_1_clip,24.00)
    np.testing.assert_equal(mask.mag_2_clip,23.98)

    # Background CMD
    cmd_background = mask.backgroundCMD(catalog)
    np.testing.assert_allclose(cmd_background[75,3:6],
                               [4.23757354, 2727.90602297,  518.12225962])

    # Catalog restriction
    sel = mask.restrictCatalogToObservableSpaceCMD(catalog)
    np.testing.assert_equal(sel.sum(),12200)
    np.testing.assert_equal(catalog.mag[sel].max() < 24.0, True)
    np.testing.assert_equal(catalog.color[sel].max() < 1.0, True)

    # Photometric errors from catalog and mask
    np.testing.assert_allclose(mask.photo_err_1(np.array([0,1,3])),
                               [0.08077547, 0.03407681, 0.006382297])
    np.testing.assert_allclose(mask.photo_err_2(np.array([0,1,3])),
                               [0.09270298, 0.03845626, 0.006904741])

if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()

    test_catalog()
    test_mask()
    test_roi()
