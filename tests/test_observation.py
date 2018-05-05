#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import numpy as np

CONFIG='ugali/config/config_test.yaml'
LON = RA = 53.92
LAT = DEC = -54.05
POINTS = np.array([(LON,LAT), (53.81,-54.02), (54.17,-53.88), (52.59,-55.39)]).T
def test_roi():
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


if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()

    test_roi()
