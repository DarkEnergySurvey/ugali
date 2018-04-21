#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import os
import ugali.utils.parser
import numpy as np

def test_targets():
    test_data = \
"""#name  lon  lat  radius coord  
object_1 354.36 -63.26 1.0 CEL
object_2 19.45  -17.46 1.0 CEL
#object_3  18.94  -41.05  1.0  CEL
"""
    with open('targets.txt','w') as f:
        f.write(test_data)
    
    parser = ugali.utils.parser.Parser()
    parser.add_coords(targets=True)
    args = parser.parse_args(['-t','targets.txt'])

    np.testing.assert_array_almost_equal(args.coords['lon'],[316.311,156.487],
                                         decimal=3)
    np.testing.assert_array_almost_equal(args.coords['lat'],[-51.903,-78.575],
                                         decimal=3)
    np.testing.assert_array_almost_equal(args.coords['radius'],[1.0,1.0],
                                         decimal=3)
    os.remove('targets.txt')
    return args

