#!/usr/bin/env python
import os,sys
from os.path import join,abspath,split
import inspect
from collections import OrderedDict as odict

import numpy as np
from numpy.lib.recfunctions import stack_arrays
import fitsio

import ugali.utils.projector
from ugali.utils.projector import gal2cel, cel2gal
import ugali.utils.idl
from ugali.utils.healpix import ang2pix
from ugali.utils.shell import get_ugali_dir, get_cat_dir
from ugali.utils.logger import logger

#class Catalog(np.recarray):
# 
#    DATADIR=os.path.join(os.path.split(os.path.abspath(__file__))[0],"../data/catalogs/")
# 
#    def __new__(cls,filename=None):
#        # Need to do it this way so that array can be resized...
#        dtype=[('name',object),
#               ('ra',float),
#               ('dec',float),
#               ('glon',float),
#               ('glat',float)]
#        self = np.recarray(0,dtype=dtype).view(cls)
#        self._load(filename)
#        return self
# 
#    def __add__(self, other):
#        return np.concatenate([self,other])
# 
#    def __getitem__(self, key):
#        """ 
#        Support indexing, slicing and direct access.
#        """
#        try:
#            return np.recarray.__getitem__(key)
#        except ValueError, message:
#            if key in self.name:
#                idx = (self.name == key)
#                return np.recarray.__getitem__(idx)
#            else:
#                raise ValueError(message)
# 
#    def _load(self,filename):
#        pass
# 
#    def match(self,lon,lat,tol=0.1,coord='gal'):
#        if coord.lower == 'cel':
#            glon, glat = ugali.utils.projector.celToGal(lon,lat)
#        else:
#            glon,glat = lon, lat
#        return ugali.utils.projector.match(glon,glat,self.data['glon'],self.data['glat'],tol)



class SourceCatalog(object):
    #join(split(abspath(__file__))[0],"../data/catalogs/") 
    DATADIR=get_cat_dir()
 
    def __init__(self, filename=None):
        columns = [('name',object),
                   ('ra',float),
                   ('dec',float),
                   ('glon',float),
                   ('glat',float)]
        self.data = np.recarray(0,dtype=columns)
        self._load(filename)
        if np.isnan([self.data['glon'],self.data['glat']]).any():
            raise ValueError("Incompatible values")
 
    def __getitem__(self, key):
        """ 
        Support indexing, slicing and direct access.
        """
        try:
            return self.data[key]
        except ValueError as message:
            if key in self.data['name']:
                return self.data[self.data['name'] == key]
            else:
                raise ValueError(message)
 
    def __add__(self, other):
        ret = SourceCatalog()
        ret.data = np.concatenate([self.data,other.data])
        return ret
        
    def __len__(self):
        """ Return the length of the collection.
        """
        return len(self.data)
 
    def _load(self,filename):
        pass
 
    def match(self,lon,lat,coord='gal',tol=0.1,nnearest=1):
        if coord.lower() == 'cel':
            glon, glat = cel2gal(lon,lat)
        else:
            glon,glat = lon, lat
        return ugali.utils.projector.match(glon,glat,self['glon'],self['glat'],tol,nnearest)
   
class McConnachie12(SourceCatalog):
    """
    Catalog of nearby dwarf spheroidal galaxies. 
    http://arxiv.org/abs/1204.1562

    https://www.astrosci.ca/users/alan/Nearby_Dwarfs_Database_files/NearbyGalaxies.dat

    """
 
    def _load(self,filename):
        if filename is None: 
            filename = os.path.join(self.DATADIR,"J_AJ_144_4/NearbyGalaxies2012.dat")
        self.filename = filename
 
        raw = np.genfromtxt(filename,delimiter=[19,3,3,5,3,3,3],usecols=range(7),dtype=['|S19']+6*[float],skip_header=36)
 
        self.data.resize(len(raw))
        self.data['name'] = np.char.strip(raw['f0'])
 
        ra = raw[['f1','f2','f3']].view(float).reshape(len(raw),-1)
        dec = raw[['f4','f5','f6']].view(float).reshape(len(raw),-1)
        self.data['ra'] = ugali.utils.projector.hms2dec(ra)
        self.data['dec'] = ugali.utils.projector.dms2dec(dec)
        
        glon,glat = cel2gal(self.data['ra'],self.data['dec'])
        self.data['glon'],self.data['glat'] = glon,glat

class McConnachie15(SourceCatalog):
    """
    Catalog of nearby dwarf spheroidal galaxies. Updated September 2015.
    http://arxiv.org/abs/1204.1562

    http://www.astro.uvic.ca/~alan/Nearby_Dwarf_Database_files/NearbyGalaxies.dat
    """
 
    def _load(self,filename):
        if filename is None: 
            filename = os.path.join(self.DATADIR,"J_AJ_144_4/NearbyGalaxies.dat")
        self.filename = filename
 
        raw = np.genfromtxt(filename,delimiter=[19,3,3,5,3,3,3],usecols=list(range(7)),dtype=['|S19']+6*[float],skip_header=36)

        self.data.resize(len(raw))
        self.data['name'] = np.char.lstrip(np.char.strip(raw['f0']),'*')

        ra = raw[['f1','f2','f3']].view(float).reshape(len(raw),-1)
        dec = raw[['f4','f5','f6']].view(float).reshape(len(raw),-1)
        self.data['ra'] = ugali.utils.projector.hms2dec(ra)
        self.data['dec'] = ugali.utils.projector.dms2dec(dec)
        
        glon,glat = cel2gal(self.data['ra'],self.data['dec'])
        self.data['glon'],self.data['glat'] = glon,glat

class Rykoff14(SourceCatalog):
    """
    Catalog of red-sequence galaxy clusters.
    http://arxiv.org/abs/1303.3562

    """

    def _load(self, filename):
        if filename is None: 
            filename = os.path.join(self.DATADIR,"redmapper/dr8_run_redmapper_v5.10_lgt20_catalog.fit")
        self.filename = filename

        raw = fitsio.read(filename,lower=True)

        self.data.resize(len(raw))
        self.data['name'] = np.char.mod("RedMaPPer %d",raw['mem_match_id'])
        self.data['ra'] = raw['ra']
        self.data['dec'] = raw['dec']
        glon,glat = cel2gal(raw['ra'],raw['dec'])
        self.data['glon'],self.data['glat'] = glon, glat

class Harris96(SourceCatalog):
    """
    Catalog of Milky Way globular clusters.
    Harris, W.E. 1996, AJ, 112, 1487

    http://physwww.physics.mcmaster.ca/~harris/mwgc.dat

    NOTE: There is some inconsistency between Equatorial and
    Galactic coordinates in the catalog. Equatorial seems more
    reliable.
    """
    def _load(self,filename):
        if filename is None: 
            filename = os.path.join(self.DATADIR,"VII_202/mwgc.dat")
        self.filename = filename

        kwargs = dict(delimiter=[12,12,3,3,6,5,3,6,8,8,6],dtype=2*['S12']+7*[float],skip_header=72,skip_footer=363)
        raw = np.genfromtxt(filename,**kwargs)

        self.data.resize(len(raw))
        self.data['name'] = np.char.strip(raw['f0'])

        ra = raw[['f2','f3','f4']].view(float).reshape(len(raw),-1)
        dec = raw[['f5','f6','f7']].view(float).reshape(len(raw),-1)

        self.data['ra'] = ugali.utils.projector.hms2dec(ra)
        self.data['dec'] = ugali.utils.projector.dms2dec(dec)

        glon,glat = cel2gal(self.data['ra'],self.data['dec'])
        self.data['glon'],self.data['glat'] = glon,glat

class Corwen04(SourceCatalog):
    """
    Modern compilation of the New General Catalogue and IC
    """
    def _load(self,filename):
        kwargs = dict(delimiter=[1,1,4,15,3,3,8,3,3,7],usecols=[1,2]+list(range(4,10)),dtype=['S1']+[int]+6*[float])
        if filename is None: 
            raw = []
            for basename in ['VII_239A/ngcpos.dat','VII_239A/icpos.dat']:
                filename = os.path.join(self.DATADIR,basename)
                raw.append(np.genfromtxt(filename,**kwargs))
            raw = np.concatenate(raw)
        else:
            raw = np.genfromtxt(filename,**kwargs)
        self.filename = filename

        # Some entries are missing...
        raw['f4'] = np.where(np.isnan(raw['f4']),0,raw['f4'])
        raw['f7'] = np.where(np.isnan(raw['f7']),0,raw['f7'])

        self.data.resize(len(raw))
        names = np.where(raw['f0'] == 'N', 'NGC %04i', 'IC %04i')
        self.data['name'] = np.char.mod(names,raw['f1'])

        ra = raw[['f2','f3','f4']].view(float).reshape(len(raw),-1)
        dec = raw[['f5','f6','f7']].view(float).reshape(len(raw),-1)
        self.data['ra'] = ugali.utils.projector.hms2dec(ra)
        self.data['dec'] = ugali.utils.projector.dms2dec(dec)

        glon,glat = cel2gal(self.data['ra'],self.data['dec'])
        self.data['glon'],self.data['glat'] = glon,glat

#class Steinicke10(SourceCatalog):
#    """
#    Another modern compilation of the New General Catalogue
#    (people still don't agree on the composition of NGC...)
#    """
#    def _load(self,filename):
#        if filename is None: 
#            filename = os.path.join(self.DATADIR,"NI2013.csv")
# 
#        raw = np.genfromtxt(filename,delimiter=',',usecols=[5,6]+range(13,20),dtype=['S1',int]+3*[float]+['S1']+3*[float])
# 
#        self.data.resize(len(raw))
#        names = np.where(raw['f0'] == 'N', 'NGC %04i', 'IC %04i')
#        self.data['name'] = np.char.mod(names,raw['f1'])
# 
#        sign = np.where(raw['f5'] == '-',-1,1)
#        ra = raw[['f2','f3','f4']].view(float).reshape(len(raw),-1)
#        dec = raw[['f6','f7','f8']].view(float).reshape(len(raw),-1)
#        dec[:,0] = np.copysign(dec[:,0], sign)
# 
#        self.data['ra'] = ugali.utils.projector.hms2dec(ra)
#        self.data['dec'] = ugali.utils.projector.dms2dec(dec)
# 
#        glon,glat = ugali.utils.projector.celToGal(self.data['ra'],self.data['dec'])
#        self.data['glon'],self.data['glat'] = glon,glat

class Nilson73(SourceCatalog):
    """
    Modern compilation of the Uppsala General Catalog

    http://vizier.cfa.harvard.edu/viz-bin/Cat?VII/26D
    """
    def _load(self,filename):
        if filename is None: 
            filename = os.path.join(self.DATADIR,"VII_26D/catalog.dat")
        self.filename = filename
        raw = np.genfromtxt(filename,delimiter=[3,7,2,4,3,2],dtype=['S3']+['S7']+4*[float])
        
        self.data.resize(len(raw))
        self.data['name'] = np.char.mod('UGC %s',np.char.strip(raw['f1']))

        ra = raw[['f2','f3']].view(float).reshape(len(raw),-1)
        ra = np.vstack([ra.T,np.zeros(len(raw))]).T
        dec = raw[['f4','f5']].view(float).reshape(len(raw),-1)
        dec = np.vstack([dec.T,np.zeros(len(raw))]).T
        ra1950 = ugali.utils.projector.hms2dec(ra)
        dec1950 = ugali.utils.projector.dms2dec(dec)
        ra2000,dec2000 = ugali.utils.idl.jprecess(ra1950,dec1950)
        self.data['ra'] = ra2000
        self.data['dec'] = dec2000

        glon,glat = cel2gal(self.data['ra'],self.data['dec'])
        self.data['glon'],self.data['glat'] = glon,glat

class Webbink85(SourceCatalog):
    """
    Structure parameters of Galactic globular clusters
    http://vizier.cfa.harvard.edu/viz-bin/Cat?VII/151

    NOTE: Includes Reticulum and some open clusters
    http://spider.seds.org/spider/MWGC/mwgc.html
    """
    def _load(self,filename):
        kwargs = dict(delimiter=[8,15,9,4,3,3,5,5],usecols=[1]+list(range(3,8)),dtype=['S13']+5*[float])
        if filename is None: 
            raw = []
            for basename in ['VII_151/table1a.dat','VII_151/table1c.dat']:
                filename = os.path.join(self.DATADIR,basename)
                raw.append(np.genfromtxt(filename,**kwargs))
            raw = np.concatenate(raw)
        else:
            raw = np.genfromtxt(filename,**kwargs)
        self.filename = filename
        
        self.data.resize(len(raw))
        #self.data['name'] = np.char.strip(raw['f0'])
        self.data['name'] = np.char.join(' ',np.char.split(raw['f0']))

        ra = raw[['f1','f2','f3']].view(float).reshape(len(raw),-1)
        dec = raw[['f4','f5']].view(float).reshape(len(raw),-1)
        dec = np.vstack([dec.T,np.zeros(len(raw))]).T
        ra1950 = ugali.utils.projector.hms2dec(ra)
        dec1950 = ugali.utils.projector.dms2dec(dec)
        ra2000,dec2000 = ugali.utils.idl.jprecess(ra1950,dec1950)
        self.data['ra'] = ra2000
        self.data['dec'] = dec2000

        glon,glat = cel2gal(self.data['ra'],self.data['dec'])
        self.data['glon'],self.data['glat'] = glon,glat


class Kharchenko13(SourceCatalog):
    """
    Global survey of star clusters in the Milky Way
    http://vizier.cfa.harvard.edu/viz-bin/Cat?J/A%2bA/558/A53

    NOTE: CEL and GAL coordinates are consistent to < 0.01 deg.
    """
    def _load(self,filename):
        kwargs = dict(delimiter=[4,18,20,8,8],usecols=[1,3,4],dtype=['S18',float,float])
        if filename is None: 
            filename = os.path.join(self.DATADIR,"J_AA_558_A53/catalog.dat")
        self.filename = filename
        raw = np.genfromtxt(filename,**kwargs)
        
        self.data.resize(len(raw))
        self.data['name'] = np.char.strip(raw['f0'])

        self.data['glon'] = raw['f1']
        self.data['glat'] = raw['f2']

        ra,dec = gal2cel(self.data['glon'],self.data['glat'])
        self.data['ra'],self.data['dec'] = ra,dec

class Bica08(SourceCatalog):
    """
    LMC star clusters
    http://cdsarc.u-strasbg.fr/viz-bin/Cat?J/MNRAS/389/678

    NOTE: CEL and GAL coordinates are consistent to < 0.01 deg.
    """
    def _load(self,filename):
        kwargs = dict(delimiter=[32,2,3,3,5,3,3],dtype=['S32']+6*[float])
        if filename is None: 
            filename = os.path.join(self.DATADIR,"J_MNRAS_389_678/table3.dat")
        self.filename = filename
        raw = np.genfromtxt(filename,**kwargs)

        self.data.resize(len(raw))
        self.data['name'] = np.char.strip(raw['f0'])
 
        ra = raw[['f1','f2','f3']].view(float).reshape(len(raw),-1)
        dec = raw[['f4','f5','f6']].view(float).reshape(len(raw),-1)
        self.data['ra'] = ugali.utils.projector.hms2dec(ra)
        self.data['dec'] = ugali.utils.projector.dms2dec(dec)
        
        glon,glat = cel2gal(self.data['ra'],self.data['dec'])
        self.data['glon'],self.data['glat'] = glon,glat

class WEBDA14(SourceCatalog):
    """
    Open cluster database.
    http://www.univie.ac.at/webda/cgi-bin/selname.cgi?auth=
    
    """
    def _load(self,filename):
        kwargs = dict(delimiter='\t',usecols=[0,1,2],dtype=['S18',float,float])
        if filename is None: 
            filename = os.path.join(self.DATADIR,"WEBDA/webda.tsv")
        self.filename = filename
        raw = np.genfromtxt(filename,**kwargs)
        
        self.data.resize(len(raw))
        self.data['name'] = np.char.strip(raw['f0'])

        self.data['glon'] = raw['f1']
        self.data['glat'] = raw['f2']

        ra,dec = gal2cel(self.data['glon'],self.data['glat'])
        self.data['ra'],self.data['dec'] = ra,dec

class ExtraDwarfs(SourceCatalog):
    """
    Collection of dwarf galaxy candidates discovered in 2015
    """
    def _load(self,filename):
        kwargs = dict(delimiter=',')
        if filename is None: 
            filename = os.path.join(self.DATADIR,"extras/extra_dwarfs.csv")
        self.filename = filename
        raw = np.recfromcsv(filename,**kwargs)
        
        self.data.resize(len(raw))
        self.data['name'] = raw['name']
        
        self.data['ra'] = raw['ra']
        self.data['dec'] = raw['dec']

        self.data['glon'],self.data['glat'] = cel2gal(raw['ra'],raw['dec'])


class ExtraClusters(SourceCatalog):
    """
    Collection of recently discovered star clusters
    """
    def _load(self,filename):
        kwargs = dict(delimiter=',')
        if filename is None: 
            filename = os.path.join(self.DATADIR,"extras/extra_clusters.csv")
        self.filename = filename
        raw = np.recfromcsv(filename,**kwargs)
        
        self.data.resize(len(raw))
        self.data['name'] = raw['name']
        
        self.data['ra'] = raw['ra']
        self.data['dec'] = raw['dec']

        self.data['glon'],self.data['glat'] = cel2gal(raw['ra'],raw['dec'])

    

def catalogFactory(name, **kwargs):
    """
    Factory for various catalogs.
    """
    fn = lambda member: inspect.isclass(member) and member.__module__==__name__
    catalogs = odict(inspect.getmembers(sys.modules[__name__], fn))

    if name not in list(catalogs.keys()):
        msg = "%s not found in catalogs:\n %s"%(name,list(kernels.keys()))
        logger.error(msg)
        msg = "Unrecognized catalog: %s"%name
        raise Exception(msg)

    return catalogs[name](**kwargs)

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('args',nargs=argparse.REMAINDER)
    opts = parser.parse_args(); args = opts.args
