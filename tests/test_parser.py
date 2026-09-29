#!/usr/bin/env python
"""
Test the ugali argument parser.
"""
__author__ = "Alex Drlica-Wagner"
import ugali.utils.parser
import numpy as np

def test_targets(tmp_path):
    test_data = \
"""#name  lon  lat  radius coord  
object_1 354.36 -63.26 1.0 CEL
object_2 19.45  -17.46 1.0 CEL
#object_3  18.94  -41.05  1.0  CEL
"""
    # Use pytest's tmp_path fixture, which is a pathlib.Path
    filename = tmp_path / 'targets.txt'
    filename.write_text(test_data)

    parser = ugali.utils.parser.Parser()
    parser.add_coords(targets=True)
    args = parser.parse_args(['-t', str(filename)])

    np.testing.assert_array_almost_equal(args.coords['lon'], [316.311, 156.487],
                                         decimal=3)
    np.testing.assert_array_almost_equal(args.coords['lat'], [-51.903, -78.575],
                                         decimal=3)
    np.testing.assert_array_almost_equal(args.coords['radius'], [1.0, 1.0],
                                         decimal=3)
    np.testing.assert_array_equal(args.names, ['object_1', 'object_2'])
