#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"

import os
from urllib import urlencode
from urllib2 import urlopen
import re
import copy
from collections import OrderedDict as odict

import numpy as np

from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

class IsochroneFile(object):
    def __init__(self,survey='des',**kwargs):
        self.survey=survey.lower()

    @classmethod
    def z2feh(cls, z):
        raise Exception("Must be implemented by child class")

    @classmethod
    def feh2z(cls, feh):
        raise Exception("Must be implemented by child class")

    @classmethod
    def params2filename(cls,age,metallicity):
        return cls._basename%dict(prefix=cls._prefix,age=age,z=metallicity)

    @classmethod
    def filename2params(cls,filename):
        #ADW: Could probably do something more clever so that parsing info
        #is stored in only one place...
        basename = os.path.basename(filename)
        prefix,a,z = os.path.splitext(basename)[0].split('_')
        if prefix != cls._prefix:
            msg = 'File prefix does not match: %s'%filename
            raise Exception(msg)
        age = float(a.strip('a'))
        metallicity = float(z.strip('z'))
        return age,metallicity

    def create_grid(self,abins=None,zbins=None):
        if abins is None and zbins is None:
            filenames = glob.glob(self.get_dirname()+'/%s_*.dat'%(self._prefix))
            data = np.array([self.filename2params(f) for f in filenames])
            if not len(data):
                msg = "No isochrone files found in: %s"%self.get_dirname()
                raise Exception(msg)
            arange = np.unique(data[:,0])
            zrange = np.unique(data[:,1])
        elif abins is not None and zbins is not None:            
            # Age in units of Gyr
            arange = np.linspace(abins[0],abins[1],abins[2]+1)
            # Metallicity sampled logarithmically
            zrange = np.logspace(np.log10(zbins[0]),np.log10(zbins[1]),zbins[2]+1)
        else:
            msg = "Must specify both `abins` and `zbins` or neither"
            raise Exception(msg)
        aa,zz = np.meshgrid(arange,zrange)
        return aa.flatten(),zz.flatten()

    def create_tree(self,grid=None):
        if grid is None: grid = self.create_grid()
        return scipy.spatial.cKDTree(np.vstack(grid).T)

    def get_filename(self):
        dirname = self.get_dirname()
        p = [self.age,self.metallicity]
        dist,idx = self.tree.query(p)
        age = self.grid[0][idx]
        z = self.grid[1][idx]
        return os.path.join(dirname,self.params2filename(age,z))

    def _cache(self,name=None):
        # For first call before init fully run
        if not hasattr(self,'tree'): return
        if name in ['distance_modulus']: return

        filename = self.get_filename()
        if filename != self.filename:
            self.filename = filename
            self._parse(self.filename)


    def print_info(self,age,metallicity):
        params = dict(age=age,z=metallicity)
        params['name'] = self.__class__.__name__
        params['survey'] = self.survey
        params['feh'] = self.isochrone.z2feh(metallicity)
        msg = 'Downloading: %(name)s (survey=%(survey)s, age=%(age).1fGyr, Z=%(z).5f, Fe/H=%(feh).3f)'%params
        logger.info(msg)
        return msg

    def query_server(self,outfile,age,metallicity):
        msg = "'query_server' not implemented by base class."
        logger.error(msg)
        raise RuntimeError(msg)

    @classmethod
    def verify(cls,filename,survey,age,metallicity):
        msg = "'verify' not implemented by base class."
        logger.error(msg)
        raise RuntimeError(msg)

    def download(self,age,metallicity,outdir=None,force=False):
        """
        Check valid parameter range and download isochrones from:
        http://stev.oapd.inaf.it/cgi-bin/cmd
        """
        if outdir is None: outdir = './'
        basename = self.isochrone.params2filename(age,metallicity)
        outfile = os.path.join(outdir,basename)
            
        if os.path.exists(outfile) and not force:
            try:
                self.verify(outfile,self.survey,age,metallicity)
                logger.info("Found %s; skipping..."%(outfile))
                return
            except Exception as e:
                msg = "Overwriting corrupted %s..."%(outfile)
                logger.warn(msg)
                #os.remove(outfile)
                
        mkdir(outdir)

        self.print_info(age,metallicity)

        try:
            self.query_server(outfile,age,metallicity)
        except Exception as e:
            logger.debug(str(e))
            raise RuntimeError('Bad server response')

        if not os.path.exists(outfile):
            raise RuntimeError('Download failed')

        try:
            self.verify(outfile,self.survey,age,metallicity)
        except Exception as e:
            msg = "Output file is corrupted."
            logger.error(msg)
            #os.remove(outfile)
            raise(e)

        return outfile

