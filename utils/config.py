"""
Class for storing and updating config dictionaries.

ADW: Can't we get rid of the 'params' member and just subclass dict?
"""
import os
import pprint
import copy
import numpy
import numpy.ma.mrecords
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
          input:   Either filename or dictionary
          default: Default configuration to merge
        
        Returns:
          config
        """
        self.update(self._load(default))
        self.update(self._load(input))

        # For back-compatibility...
        self.params = self

        # Possible filenames from this config (masked by existence)
        self.filenames = self.getFilenames()

    def _load(self, input):
        if isinstance(input, basestring):
            ext = os.path.splitext(input)[1]
            if ext == '.py':
                reader = open(input)
                params = eval(''.join(reader.readlines()))
                reader.close()
            elif ext == '.yaml':
                params = yaml.load(open(input))
            else:
                raise Exception('Unrecognized config format: %s'%ext)
        elif isinstance(input, dict):
            params = input
        elif input is None:
            params = {}
        else:
            raise Exception('Unrecognized input')

        return params

    def write(self, outfile):
        ext = os.path.splitext(outfile)[1]
        writer = open(outfile, 'w')
        if ext == '.py':
            writer.write(pprint.pformat(self.params))
        elif ext == '.yaml':
            writer.write(yaml.dump(self.params))
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
        nside_catalog = self.params['coords']['nside_catalog']

        if nside_catalog is None:
            pixels = [None]
        elif pixels is not None:
            pixels = [pixels] if numpy.isscalar(pixels) else pixels
        else:
            pixels = numpy.arange(healpy.nside2npix(nside_catalog))   

        npix = len(pixels)

        catalog_dir = self.params['catalog']['dirname']
        catalog_base = self.params['catalog']['basename']
         
        mask_dir = self.params['mask']['dirname']
        mask_base_1 = self.params['mask']['basename_1']
        mask_base_2 = self.params['mask']['basename_2']
         
        data = numpy.ma.empty(npix,dtype=[('pix',int), ('catalog',object), 
                                          ('mask_1',object), ('mask_2',object)])
        mask = numpy.ma.empty(npix,dtype=[('pix',bool), ('catalog',bool), 
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
        mask['pix'] = mask['catalog'] | mask['mask_1'] | mask['mask_2']
        #return numpy.ma.mrecords.MaskedArray(data, mask, fill_value=[-1,None,None,None])
        return numpy.ma.mrecords.MaskedArray(data, mask, fill_value=[-1,'','',''])


############################################################
