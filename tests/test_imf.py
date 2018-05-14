#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import numpy as np

import ugali.analysis.imf
import ugali.isochrone

MASSES = np.array([0.1, 0.5, 1.0, 10.0])
SEED = 1

def test_imf():
    dn_dm = ugali.analysis.imf.chabrierIMF(MASSES)
    np.testing.assert_allclose(dn_dm, 
                               [1.29919665, 0.66922584, 0.3666024, 0.01837364],
                               rtol=1e-6)
                               
    #imf = ugali.analysis.imf.IMF('chabrier')
    imf = ugali.analysis.imf.factory('Chabrier2003')
    masses = imf.sample(3, seed=SEED)
    np.testing.assert_allclose(masses,[0.22122041, 0.4878909, 0.10002029],rtol=1e-6)

    imf = ugali.analysis.imf.Chabrier2003()
    masses = imf.sample(3, seed=SEED)
    np.testing.assert_allclose(masses,[0.22122041, 0.4878909, 0.10002029],rtol=1e-6)

    iso = ugali.isochrone.Bressan2012(imf_type='Chabrier2003')
    masses = iso.imf.sample(3,seed=SEED)
    np.testing.assert_allclose(masses,[0.22122041, 0.4878909, 0.10002029],rtol=1e-6)

    integral = iso.imf.integrate(0.1,2.0)
    np.testing.assert_allclose(integral,0.94961632593)

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    
    test_imf()
