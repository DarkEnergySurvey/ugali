#!/usr/bin/env python
"""
Module for dealing with Isochrones.
"""
from ugali.isochrone.model import get_iso_dir
from ugali.isochrone.composite import factory
from ugali.isochrone.composite import CompositeIsochrone, Padova, Dotter
from ugali.isochrone.parsec import Bressan2012, Marigo2017
from ugali.isochrone.dartmouth import Dotter2008
from ugali.isochrone.mesa import Dotter2016

