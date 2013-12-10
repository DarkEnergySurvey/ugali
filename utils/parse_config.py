"""
Class for storing and updating config dictionaries.
"""
import os
import pprint
import copy
import numpy
import numpy.ma.mrecords
import healpy

import ugali.utils.parse_config # To recognize own type

############################################################

class Config(object):
    """
    Documentation.
    """

    def __init__(self, input = None):
        """
        Initialize the Config object with a variety of possible input formats.

        INPUTS:
            input[None]: can be a filename, parameter dictionary, or another Config object
        """
        if type(input) is str:
            reader = open(input)
            self.params = eval(''.join(reader.readlines()))
            reader.close()
        elif type(input) is dict:
            self.params = input
        elif type(input) is ugali.utils.parse_config.Config:
            self.params = input.params
        else:
            self.params = {}

        # Possible filenames from this config (masked by existence)
        self.filenames = self.getFilenames()

    def merge(self, merge, overwrite = False):
        """
        Update parameters from a second param dictionary or Config object.

        INPUTS
            merge: a second param dictionary or Config object
            overwrite[False]: overwrite the parameter dictionary
        RETURNS
            merged parameter dictionary
        """
        if type(merge) is dict:
            params = self._mergeParams(merge)
        elif type(merge) is ugali.utils.parse_config.Config:
            params = self._mergeParams(merge.params)
        elif merge is None:
            params = self.params
        else:
            print 'WARNING: did not recognize %s'%(type(merge))
            params = self.params

        if overwrite:
            self.params = params

        return params
    
    def writeConfig(self, config_outfile):
        writer = open(config_outfile, 'w')
        writer.write(pprint.pformat(self.params))
        writer.close()

    def show(self):
        print pprint.pformat(self.params)

    def _mergeParams(self, params_update):
        """
        Helper function to merge parameters from two dictionaries.
        """
        params_merge = copy.copy(self.params)
        
        for section in params_update.keys():
            if section not in params_merge.keys():
                params_merge[section] = {}
                if type(params_update[section]) is dict:
                    for key in params_update[section].keys():
                        params_merge[section][key] = params_update[section][key]
                        
        return params_merge

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

        catalog_dir = self.params['catalog']['dirname']
        catalog_base = self.params['catalog']['basename']
         
        mask_dir = self.params['mask']['dirname']
        mask_base_1 = self.params['mask']['basename_1']
        mask_base_2 = self.params['mask']['basename_2']
         
        data = numpy.ma.empty(len(pixels), dtype=[('pix', int),('catalog', object), 
                                                       ('mask_1', object),('mask_2', object)])
        mask = numpy.ma.empty(len(pixels), dtype=[('pix', bool),('catalog', bool), 
                                                       ('mask_1', bool),('mask_2', bool)])
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
