"""
Class for storing and updating config dictionaries.
"""
import os,sys
from os.path import join, exists
import pprint
import copy
from collections import OrderedDict as odict
import glob

import numpy as np
import healpy as hp

from ugali.utils.logger import logger
import ugali.utils.config # To recognize own type
from ugali.utils.mlab import isstring

try: import yaml
except ImportError: logger.warning("YAML not found")

class Config(dict):
    """
    Configuration object
    """

    def __init__(self, config, default=None):
        """
        Initialize a configuration object from a filename or a dictionary.
        Provides functionality to merge with a default configuration.

        Parameters:
          config:   filename, dict, or Config object (deep copied)
          default:  default configuration to merge
        
        Returns:
          config
        """
        self.update(self._load(default))
        self.update(self._load(config))

        self._formatFilepaths()

        # For back-compatibility...
        self.params = self

        # Run some basic validation
        # ADW: This should be run after creating filenames
        self._validate()

        # Filenames from this config (masked by existence) 
        # ADW: We should not recreate filenames if they already exist
        # in the input config
        if not hasattr(self,'filenames'):
            try:
                self.filenames = self._createFilenames()
            except:
                exc_type,exc_value,exc_traceback = sys.exc_info()
                logger.warning("%s %s"%(exc_type,exc_value))
                logger.warning("Filenames could not be created for config.")

    def __str__(self):
        return yaml.dump(self)

    def _load(self, config):
        """ Load this config from an existing config 

        Parameters:
        -----------
        config : filename, config object, or dict to load

        Returns:
        --------
        params : configuration parameters
        """
        if isstring(config):
            self.filename = config
            params = yaml.load(open(config))
        elif isinstance(config, Config):
            # This is the copy constructor...
            self.filename = config.filename
            params = copy.deepcopy(config)
        elif isinstance(config, dict):
            params = copy.deepcopy(config)
        elif config is None:
            params = {}
        else:
            raise Exception('Unrecognized input')

        return params

    def _validate(self):
        """ Enforce some structure to the config file """
        # This could be done with a default config

        # Check that specific keys exist
        sections = odict([
                ('catalog',['dirname','basename',
                            'lon_field','lat_field','objid_field',
                            'mag_1_band', 'mag_1_field', 'mag_err_1_field',
                            'mag_2_band', 'mag_2_field', 'mag_err_2_field',
                            ]),
                ('mask',[]),
                ('coords',['nside_catalog','nside_mask','nside_likelihood',
                           'nside_pixel','roi_radius','roi_radius_annulus',
                           'roi_radius_interior','coordsys',
                           ]),
                ('likelihood',[]),
                ('output',[]),
                ('batch',[]),
                ])  

        keys = np.array(list(sections.keys()))
        found = np.in1d(keys,list(self.keys()))

        if not np.all(found):
            msg = 'Missing sections: '+str(keys[~found])
            raise Exception(msg)

        for section,keys in sections.items():
            keys = np.array(keys)
            found = np.in1d(keys,list(self[section].keys()))
            if not np.all(found):
                msg = 'Missing keys in %s: '%(section)+str(keys[~found])
                raise Exception(msg)

        #if hasattr(self,'filenames'):
        #    len(self.filenames)
            
    def _formatFilepaths(self):
        """
        Join dirnames and filenames from config.
        """
        likedir=self['output']['likedir']
        self.likefile  = join(likedir,self['output']['likefile'])
        self.mergefile = join(likedir,self['output']['mergefile'])
        self.roifile   = join(likedir,self['output']['roifile'])

        searchdir=self['output']['searchdir']
        self.labelfile  = join(searchdir,self['output']['labelfile'])
        self.objectfile = join(searchdir,self['output']['objectfile'])
        self.assocfile  = join(searchdir,self['output']['assocfile'])
        self.candfile   = join(searchdir,self['output']['candfile'])

        mcmcdir=self['output']['mcmcdir']
        self.mcmcfile   = join(mcmcdir,self['output']['mcmcfile'])

    def write(self, filename):
        """
        Write a copy of this config object.

        Parameters:
        -----------
        outfile : output filename
        
        Returns:
        --------
        None
        """
        ext = os.path.splitext(filename)[1]
        writer = open(filename, 'w')
        if ext == '.py':
            writer.write(pprint.pformat(self))
        elif ext == '.yaml':
            writer.write(yaml.dump(self))
        else:
            writer.close()
            raise Exception('Unrecognized config format: %s'%ext)
        writer.close()

    def _createFilenames(self,pixels=None):
        """
        Create a masked records array of all filenames for the given set of
        pixels and store the existence of those files in the mask values.

        Examples:
        f = getFilenames([1,2,3])
        # All possible catalog files
        f['catalog'].data
        # All existing catalog files
        f['catalog'][~f.mask['catalog']]
        # or
        f['catalog'].compressed()
        # All missing mask_1 files
        f['mask_1'][f.mask['mask_1']]
        # Pixels where all files exist
        f['pix'][~f.mask['pix']]

        Parameters:
        -----------
        pixels : If pixels is None, grab all pixels of 'nside_catalog'.

        Returns:
        --------
        recarray : pixels and mask value
        """
        nside_catalog = self['coords']['nside_catalog']

        # Deprecated: ADW 2018-06-17
        #if nside_catalog is None:
        #    pixels = [None]
        if pixels is not None:
            pixels = [pixels] if np.isscalar(pixels) else pixels
        else:
            pixels = np.arange(hp.nside2npix(nside_catalog))   

        npix = len(pixels)

        catalog_dir = self['catalog']['dirname']
        catalog_base = self['catalog']['basename']
         
        mask_dir = self['mask']['dirname']
        mask_base_1 = self['mask']['basename_1']
        mask_base_2 = self['mask']['basename_2']
         
        data = np.ma.empty(npix,dtype=[('pix',int), ('catalog',object), 
                                       ('mask_1',object), ('mask_2',object)])
        mask = np.ma.empty(npix,dtype=[('pix',bool), ('catalog',bool), 
                                       ('mask_1',bool), ('mask_2',bool)])
        for ii,pix in enumerate(pixels):
            if pix is None:
                # DEPRECTATED: ADW 2018-06-17
                # This is not really being used anymore
                catalog = os.path.join(catalog_dir,catalog_base)
                mask_1 = os.path.join(mask_dir,mask_base_1)
                mask_2 = os.path.join(mask_dir,mask_base_2)
            else:
                catalog = os.path.join(catalog_dir,catalog_base%pix)
                mask_1 = os.path.join(mask_dir,mask_base_1%pix)
                mask_2 = os.path.join(mask_dir,mask_base_2%pix)
            data[ii]['pix'] = pix if pix is not None else -1
            data[ii]['catalog'] = catalog
            data[ii]['mask_1']  = mask_1
            data[ii]['mask_2']  = mask_2
         
            mask[ii]['catalog'] = not os.path.exists(catalog)
            mask[ii]['mask_1']  = not os.path.exists(mask_1)
            mask[ii]['mask_2']  = not os.path.exists(mask_2)

        for name in ['catalog','mask_1','mask_2']:
            if np.all(mask[name]): logger.warn("All '%s' files masked"%name)

        # mask 'pix' if all files not present
        mask['pix'] = mask['catalog'] | mask['mask_1'] | mask['mask_2']

        if np.all(mask['pix']): logger.warn("All pixels masked")
                
        #return np.ma.mrecords.MaskedArray(data, mask, fill_value=[-1,None,None,None])
        #return np.ma.mrecords.MaskedArray(data, mask, fill_value=[-1,'','',''])
        return np.ma.MaskedArray(data, mask, fill_value=[-1,'','',''])


    def _createFilenames(self):
        """
        Create a masked records array of all filenames for the given set of
        pixels and store the existence of those files in the mask values.

        Parameters:
        -----------
        None

        Returns:
        --------
        recarray : pixels and mask value
        """
        nside_catalog = self['coords']['nside_catalog']
        npix = hp.nside2npix(nside_catalog)
        pixels = np.arange(npix)

        catalog_dir = self['catalog']['dirname']
        catalog_base = self['catalog']['basename']
        catalog_path = os.path.join(catalog_dir,catalog_base)

        mask_dir    = self['mask']['dirname']
        mask_base_1 = self['mask']['basename_1']
        mask_base_2 = self['mask']['basename_2']
        mask_path_1 = os.path.join(mask_dir,mask_base_1)
        mask_path_2 = os.path.join(mask_dir,mask_base_2)

        data = np.ma.empty(npix,dtype=[('pix',int), ('catalog',object), 
                                       ('mask_1',object), ('mask_2',object)])
        mask = np.ma.empty(npix,dtype=[('pix',bool), ('catalog',bool), 
                                       ('mask_1',bool), ('mask_2',bool)])

        # Build the filenames
        data['pix']     = pixels
        data['catalog'] = np.char.mod(catalog_path,pixels)
        data['mask_1']  = np.char.mod(mask_path_1,pixels)
        data['mask_2']  = np.char.mod(mask_path_2,pixels)

        # Build the mask of existing files using glob
        mask['catalog'] = ~np.in1d(data['catalog'],glob.glob(catalog_dir+'/*'))
        mask['mask_1']  = ~np.in1d(data['mask_1'],glob.glob(mask_dir+'/*'))
        mask['mask_2']  = ~np.in1d(data['mask_2'],glob.glob(mask_dir+'/*'))

        for name in ['catalog','mask_1','mask_2']:
            if np.all(mask[name]): logger.warn("All '%s' files masked"%name)

        # mask 'pix' if all files not present
        mask['pix'] = mask['catalog'] | mask['mask_1'] | mask['mask_2']

        if np.all(mask['pix']): logger.warn("All pixels masked")

        return np.ma.MaskedArray(data, mask, fill_value=[-1,'','',''])

    def getFilenames(self,pixels=None):
        """
        Return the requested filenames.

        Parameters:
        -----------
        pixels : requeseted pixels

        Returns:
        --------
        filenames : recarray
        """
        logger.debug("Getting filenames...")
        if pixels is None:
            return self.filenames
        else:
            return self.filenames[np.in1d(self.filenames['pix'],pixels)]

        
    getCatalogFiles = getFilenames

############################################################
