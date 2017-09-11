#!/usr/bin/env python
"""
Generic python script.
"""
import os

from ugali.isochrone.parsec import PadovaIsochrone

class EmpiricalPadova(PadovaIsochrone):
    _prefix = 'iso'
    _basename = '%(prefix)s_a13.7_z0.00007.dat'
    _dirname =  os.path.join(get_iso_dir(),'{survey}','empirical')

    defaults = (PadovaIsochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
    )

class M92(EmpiricalPadova):
    """ Empirical isochrone derived from the M92 ridgeline dereddened
    and transformed to the DES system.
    """
    _params = odict([
        ('distance_modulus', Parameter(15.0, [10.0, 30.0]) ),
        ('age',              Parameter(13.7, [13.7, 13.7]) ),  # Gyr
        ('metallicity',      Parameter(7e-5,[7e-5,7e-5]) ),
    ])

    _prefix = 'm92'
    _basename = '%(prefix)s_a13.7_z0.00007.dat'

class DESDwarfs(EmpiricalPadova):
    """ Empirical isochrone derived from spectroscopic members of the
    DES dwarfs.
    """
    _params = odict([
        ('distance_modulus', Parameter(15.0, [10.0, 30.0]) ),
        ('age',              Parameter(12.5, [12.5, 12.5]) ),  # Gyr
        ('metallicity',      Parameter(1e-4, [1e-4,1e-4]) ),
    ])

    _prefix = 'dsph'
    _basename = '%(prefix)s_a12.5_z0.00010.dat'
