#!/usr/bin/env python
"""
Testing healpix module
"""
__author__ = "Alex Drlica-Wagner"

import unittest

import numpy as np
import healpy as hp
import fitsio

from ugali.utils import healpix
from ugali.utils.logger import logger
logger.setLevel(logger.WARN)

NSIDE = 4096
FACTOR = 4
PIX = np.array([104582830,  43361203, 142027178])
U_GRADE_PIX_NEST = np.array([[418331320, 418331321, 418331322, 418331323],
                             [173444812, 173444813, 173444814, 173444815],
                             [568108712, 568108713, 568108714, 568108715]])
U_GRADE_PIX_RING = np.array([[418356572, 418323804, 418323803, 418291036],
                             [173492070, 173459302, 173459301, 173426534],
                             [568152916, 568120148, 568120147, 568087380]])

D_GRADE_PIX_NEST = PIX//FACTOR
D_GRADE_PIX_RING = np.array([26142551, 10842585, 35509461])


UD_GRADE_PIX = np.repeat(PIX,4).reshape(-1,4)
DU_GRADE_PIX_NEST = np.array([[104582828, 104582829, 104582830, 104582831],
                              [ 43361200,  43361201,  43361202,  43361203],
                              [142027176, 142027177, 142027178, 142027179]])
DU_GRADE_PIX_RING = np.array([[104582830, 104566446, 104566445, 104550062],
                              [ 43393971,  43377587,  43377586,  43361203],
                              [142059946, 142043562, 142043561, 142027178]])



class TestHealpix(unittest.TestCase):
    """Test healpix module"""

    def test_ud_grade_ipix(self):

        # Same NSIDE
        np.testing.assert_equal(healpix.ud_grade_ipix(PIX,NSIDE,NSIDE),PIX)

        # Increase resolution (u_grade)
        np.testing.assert_equal(healpix.ud_grade_ipix(PIX,NSIDE,NSIDE*2,nest=True),
                                U_GRADE_PIX_NEST)
        np.testing.assert_equal(healpix.ud_grade_ipix(PIX,NSIDE,NSIDE*2),
                                U_GRADE_PIX_RING) 

        # Decrease resolution (d_grade)
        np.testing.assert_equal(healpix.ud_grade_ipix(PIX,NSIDE,NSIDE//2,nest=True),
                                D_GRADE_PIX_NEST)
        np.testing.assert_equal(healpix.ud_grade_ipix(PIX,NSIDE,NSIDE//2),
                                D_GRADE_PIX_RING)


        # u_grade then d_grade
        np.testing.assert_equal(healpix.ud_grade_ipix(
             healpix.ud_grade_ipix(PIX,NSIDE,NSIDE*2,nest=True),
             NSIDE*2, NSIDE, nest=True),
             UD_GRADE_PIX)
        np.testing.assert_equal(healpix.ud_grade_ipix(
             healpix.ud_grade_ipix(PIX,NSIDE,NSIDE*2),NSIDE*2, NSIDE),
             UD_GRADE_PIX)

        # d_grade then u_grade
        np.testing.assert_equal(healpix.ud_grade_ipix(
            healpix.ud_grade_ipix(PIX, NSIDE, NSIDE//2, nest=True),
            NSIDE//2, NSIDE, nest=True),
            DU_GRADE_PIX_NEST
        )
        np.testing.assert_equal(healpix.ud_grade_ipix(
            healpix.ud_grade_ipix(PIX,NSIDE,NSIDE//2),NSIDE//2, NSIDE),
            DU_GRADE_PIX_RING
        )



