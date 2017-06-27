"""
Classes which manage object catalogs live here.
"""
import numpy as np
import fitsio
import copy
from matplotlib import mlab

import ugali.utils.projector

from ugali.utils.config import Config
from ugali.utils.projector import gal2cel,cel2gal
from ugali.utils.healpix import ang2pix,superpixel
from ugali.utils.logger import logger
from ugali.utils.fileio import load_infiles

class Catalog:

    def __init__(self, config, roi=None, data=None, filenames=None):
        """
        Class to store information about detected objects. This class
        augments the raw data array with several aliases and derived
        quantities.

        Parameters:
        -----------
        config    : Configuration object
        roi       : Region of Interest to load catalog data for
        data      : Data array object
        filenames : FITS filenames to read catalog from

        Returns:
        --------
        catalog   : The Catalog object
        """
        self.config = Config(config)

        if data is None:
            self._parse(roi,filenames)
        else:
            self.data = data

        self._defineVariables()

    def __add__(self, other):
        return mergeCatalogs([self,other])

    def __len__(self):
        return len(self.objid)

    def applyCut(self, cut):
        """
        Return a new catalog which is a subset of objects selected
        using the input cut array.

        NOTE: This should really be a selection.
        """
        return Catalog(self.config, data=self.data[cut])

    def bootstrap(self, mc_bit=0x10, seed=None):
        """
        Return a random catalog by boostrapping the colors of the objects in the current catalog.
        """
        if seed is not None: np.random.seed(seed)
        data = copy.deepcopy(self.data)
        idx = np.random.randint(0,len(data),len(data))
        data[self.config['catalog']['mag_1_field']][:] = self.mag_1[idx]
        data[self.config['catalog']['mag_err_1_field']][:] = self.mag_err_1[idx]
        data[self.config['catalog']['mag_2_field']][:] = self.mag_2[idx]
        data[self.config['catalog']['mag_err_2_field']][:] = self.mag_err_2[idx]
        data[self.config['catalog']['mc_source_id_field']][:] |= mc_bit
        return Catalog(self.config, data=data)

    def project(self, projector = None):
        """
        Project coordinates on sphere to image plane using Projector class.
        """
        if projector is None:
            try:
                self.projector = ugali.utils.projector.Projector(self.config['coords']['reference'][0],
                                                                 self.config['coords']['reference'][1])
            except KeyError:
                logger.warning('Projection reference point is median (lon, lat) of catalog objects')
                self.projector = ugali.utils.projector.Projector(np.median(self.lon), np.median(self.lat))
        else:
            self.projector = projector

        self.x, self.y = self.projector.sphereToImage(self.lon, self.lat)

    def spatialBin(self, roi):
        """
        Calculate indices of ROI pixels corresponding to object locations.
        """
        if hasattr(self,'pixel_roi_index') and hasattr(self,'pixel'): 
            logger.warning('Catalog alread spatially binned')
            return

        # ADW: Not safe to set index = -1 (since it will access last entry); 
        # np.inf would be better...
        self.pixel = ang2pix(self.config['coords']['nside_pixel'],self.lon,self.lat)
        self.pixel_roi_index = roi.indexROI(self.lon,self.lat)

        if np.any(self.pixel_roi_index < 0):
            logger.warning("Objects found outside ROI")

    def write(self, outfile, clobber=True, **kwargs):
        """
        Write the current object catalog to FITS file.

        Parameters:
        -----------
        filename : the FITS file to write.
        clobber  : remove existing file
        kwargs   : passed to fitsio.write

        Returns:
        --------
        None
        """
        fitsio.write(outfile,self.data,clobber=True,**kwargs)

    def _parse(self, roi=None, filenames=None):
        """        
        Parse catalog FITS files into recarray.

        Parameters:
        -----------
        roi : The region of interest; if 'roi=None', read all catalog files

        Returns:
        --------
        None
        """
        if (roi is not None) and (filenames is not None):
            msg = "Cannot take both roi and filenames"
            raise Exception(msg)

        if roi is not None:
            pixels = roi.getCatalogPixels()
            filenames = self.config.getFilenames()['catalog'][pixels]
        elif filenames is None:
            filenames = self.config.getFilenames()['catalog'].compressed()
        else:
            filenames = np.atleast_1d(filenames)

        if len(filenames) == 0:
            msg = "No catalog files found."
            raise Exception(msg)

        # Load the data
        self.data = load_infiles(filenames)

        # Apply a selection cut
        self._applySelection()

        # Cast data to recarray (historical reasons)
        self.data = self.data.view(np.recarray)

    def _applySelection(self,selection=None):
        # ADW: This is a hack (eval is unsafe!)
        if selection is None:
            selection = self.config['catalog'].get('selection')

        if not selection: 
            pass
        elif 'self.data' not in selection:
            msg = "Selection does not contain 'data'"
            raise Exception(msg)
        else:
            logger.warning('Evaluating selection: \n"%s"'%selection)
            sel = eval(selection)
            self.data = self.data[sel]
        
    def _defineVariables(self):
        """
        Helper funtion to define pertinent variables from catalog data.

        ADW (20170627): This has largely been replaced by properties.
        """
        logger.info('Catalog contains %i objects'%(len(self.data)))

        mc_source_id_field = self.config['catalog']['mc_source_id_field']
        if mc_source_id_field is not None:
            if mc_source_id_field not in self.data.dtype.names:
                array = np.zeros(len(self.data),dtype=int)
                self.data = mlab.rec_append_fields(self.data,
                                                   names=mc_source_id_field,
                                                   arrs=array)
            logger.info('Found %i simulated objects'%(np.sum(self.mc_source_id>0)))

    # Use properties to avoid duplicating the data
    @property
    def objid(self): return self.data[self.config['catalog']['objid_field']]
    @property
    def lon(self): return self.data[self.config['catalog']['lon_field']]
    @property 
    def lat(self): return self.data[self.config['catalog']['lat_field']]

    @property
    def mag_1(self): return self.data[self.config['catalog']['mag_1_field']]
    @property
    def mag_err_1(self):
        return self.data[self.config['catalog']['mag_err_1_field']]
    @property
    def mag_2(self):
        return self.data[self.config['catalog']['mag_2_field']]
    @property
    def mag_err_2(self):
        return self.data[self.config['catalog']['mag_err_2_field']]
    @property 
    def mag(self):
        if self.config['catalog']['band_1_detection']: return self.mag_1
        else: return self.mag_2
    @property
    def mag_err(self):
        if self.config['catalog']['band_1_detection']: return self.mag_err_1
        else: return self.mag_err_2
    @property
    def color(self): return self.mag_1 - self.mag_2
    @property
    def color_err(self): return np.sqrt(self.mag_err_1**2 + self.mag_err_2**2)
    @property
    def mc_source_id(self):
        return self.data.field(self.config['catalog']['mc_source_id_field'])

    # This assumes Galactic coordinates
    @property
    def ra_dec(self): return gal2cel(self.lon,self.lat)
    @property
    def ra(self): return self.ra_dec[0]
    @property
    def dec(self): return self.ra_dec[1]

    @property
    def glon_glat(self): return self.lon,self.lat
    @property
    def glon(self): return self.lon
    @property
    def glat(self): return self.lat

############################################################

def mergeCatalogs(catalog_list):
    """
    Merge a list of Catalogs.

    Parameters:
    -----------
    catalog_list : List of Catalog objects.

    Returns:
    --------
    catalog      : Combined Catalog object 
    """
    config = catalog_list[0]
    # Check the columns
    for c in catalog_list:
        if c.data.dtype.names != catalog_list[0].data.dtype.names:
            msg = "Catalog data columns not the same."
            raise Exception(msg)
    data = np.concatenate([c.data for c in catalog_list])
    return Catalog(config,data=data)
