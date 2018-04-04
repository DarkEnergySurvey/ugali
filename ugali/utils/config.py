"""
Class for storing and updating config dictionaries.
"""
import os,sys
from os.path import join, exists
import pprint
import copy
import numpy as np
import healpy

from ugali.utils.logger import logger
import ugali.utils.config # To recognize own type

try: import yaml
except ImportError: logger.warning("YAML not found")

class Config(dict):
    """
    Configuration object
    """

    def __init__(self, input, default=None):
        """
        Initialize a configuration object from a filename or a dictionary.
        Provides functionality to merge with a default configuration.

        Parameters:
          input:   Either filename or dictionary (deep copied)
          default: Default configuration to merge
        
        Returns:
          config
        """
        self.update(self._load(default))
        self.update(self._load(input))

        # For back-compatibility...
        self.params = self

        # Possible filenames from this config (masked by existence)
        try:
            self.filenames = self.getFilenames()
            self._makeFilenames()
        except:
            exc_type,exc_value,exc_traceback = sys.exc_info()
            logger.warning("%s %s"%(exc_type,exc_value))
            logger.warning("Filenames could not be created for config.")

    def __str__(self):
        return yaml.dump(self)

    def _load(self, input):
        if isinstance(input, basestring):
            self.filename = input
            ext = os.path.splitext(input)[1]
            if ext == '.py':
                # ADW: This is dangerous and terrible!!!
                # THIS SHOULD BE DEPRICATED!!!
                reader = open(input)
                params = eval(''.join(reader.readlines()))
                reader.close()
            elif ext == '.yaml':
                params = yaml.load(open(input))
            else:
                raise Exception('Unrecognized config format: %s'%ext)
        elif isinstance(input, Config):
            # This is the copy constructor...
            self.filename = input.filename
            params = copy.deepcopy(input)
        elif isinstance(input, dict):
            params = copy.deepcopy(input)
        elif input is None:
            params = {}
        else:
            raise Exception('Unrecognized input')

        return params

    def _makeFilenames(self):
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

    def write(self, outfile):
        ext = os.path.splitext(outfile)[1]
        writer = open(outfile, 'w')
        if ext == '.py':
            writer.write(pprint.pformat(self))
        elif ext == '.yaml':
            writer.write(yaml.dump(self))
        else:
            writer.close()
            raise Exception('Unrecognized config format: %s'%ext)
        writer.close()

    def getFilenames(self,pixels=None):
        """
        Create a masked records array of all filenames for the given set of
        pixels and store the existence of those files in the mask values.
        If pixels is None, default behavior is to try to join grab
        dirname + basename with no pixel insertion. If pixels == -1, grab
        all pixels of 'nside_catalog'.

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
        """
        nside_catalog = self['coords']['nside_catalog']

        if nside_catalog is None:
            pixels = [None]
        elif pixels is not None:
            pixels = [pixels] if np.isscalar(pixels) else pixels
        else:
            pixels = np.arange(healpy.nside2npix(nside_catalog))   

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

        # 'pix' is masked if all files not present
        mask['pix'] = mask['catalog'] | mask['mask_1'] | mask['mask_2']

        if np.all(mask['pix']): logger.warn("All pixels masked")
                

        #return np.ma.mrecords.MaskedArray(data, mask, fill_value=[-1,None,None,None])
        return np.ma.mrecords.MaskedArray(data, mask, fill_value=[-1,'','',''])

    getCatalogFiles = getFilenames

############################################################
