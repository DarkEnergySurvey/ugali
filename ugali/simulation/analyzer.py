#!/usr/bin/env python
"""
Analyze a simulated data set.
"""
__author__ = "Alex Drlica-Wagner"
import copy
import os

import numpy
import numpy as np
import scipy.interpolate
import healpy
import numpy.lib.recfunctions as recfuncs
import fitsio
from matplotlib import mlab

import ugali.observation.catalog
import ugali.observation.mask
import ugali.observation.roi
import ugali.utils.projector
import ugali.utils.stats
import ugali.analysis.scan

from ugali.utils.projector import gal2cel, cel2gal, sr2deg, mod2dist
from ugali.utils.healpix import ang2pix, pix2ang
from ugali.utils.logger import logger
from ugali.utils.config import Config

class Analyzer(object):
    """
    Class for analyzing simulated data
    """
    def __init__(self, config, catfile=None, popfile=None):
        self.config = Config(config)
        self.catalog = self.read_catalog(catfile)
        self.population = self.read_population(popfile)
                
    def read_population(self, filename=None):
        if not filename:
            filename = os.path.join(self.config['simulate']['dirname'],self.config['simulate']['popfile'])
        pop = ugali.utils.fileio.read(filename)
        pop.dtype.names = list(map(str.upper,pop.dtype.names))
        return pop

    def read_catalog(self, filename=None):
        if not filename:
            filename = os.path.join(self.config['simulate']['dirname'],self.config['simulate']['catfile'])
        catalog =  ugali.observation.catalog.Catalog(self.config,filenames=filename)
        catalog.data = mlab.rec_append_fields(catalog.data,
                                           names=['PIX8','PIX4096'],
                                           arrs=np.zeros((2,len(catalog.lon)),dtype='>i8'))
        return catalog

    def run(self, population=None, catalog=None, outfile=None, mc_source_id=None):
        if not population: population = self.population
        if not catalog:    catalog = self.catalog
        
        # Select only systems that are in the catalog
        sel = np.in1d(population['MC_SOURCE_ID'],catalog.mc_source_id)
        population = population[sel]

        if mc_source_id is not None:
            sel = np.in1d(population['MC_SOURCE_ID'],mc_source_id)
            if not sel.sum():
                msg = "Requested MC_SOURCE_ID not found: %i"%mc_source_id
                logger.warn(msg)
                return
            population = population[sel]

            
        size = len(population)
        dtype=[('KERNEL','S18'),('TS','>f4'),('FIT_KERNEL','S18'),('FIT_TS','>f4'),
               ('FIT_MASS','>f4'),('FIT_MASS_ERR','>f4'),
               ('FIT_DISTANCE','>f4'),('FIT_DISTANCE_ERR','>f4')]
        results = np.array(np.nan*np.ones(size),dtype=dtype)
        results = recfuncs.merge_arrays([population,results],flatten=True,asrecarray=False,usemask=False)
        self.results = results

        if outfile: 
            ugali.utils.fileio.write(outfile,results,clobber=True)

        for i,d in enumerate(results): 
            params = dict(list(zip(results.dtype.names,d)))
            lon,lat = params['RA'],params['DEC']
            distance_modulus = params['DISTANCE_MODULUS']
            mc_source_id = params['MC_SOURCE_ID']
            logger.info('\n(%i/%i); (id, lon, lat) = (%i, %.2f, %.2f)'%(i+1,size,mc_source_id,lon,lat))

            logger.info("Reading data catalog...")
            source = ugali.analysis.loglike.createSource(self.config,lon=lon,lat=lat)
            obs = ugali.analysis.loglike.createObservation(self.config,lon=lon,lat=lat)

            # Select just the simulated target of interest
            logger.info("Merging simulated catalog...")
            data = catalog.data[catalog.mc_source_id == mc_source_id].copy()
            data = np.array(data[list(obs.catalog.data.dtype.names)],dtype=obs.catalog.data.dtype)
            obs.catalog = ugali.observation.catalog.Catalog(self.config, data=np.concatenate([obs.catalog.data,data]))

            # Index of closest distance modulus
            loglike = ugali.analysis.loglike.LogLikelihood(self.config,obs,source)
            grid    = ugali.analysis.scan.GridSearch(self.config,loglike)

            self.grid = grid
            self.loglike = self.grid.loglike

            pix = self.loglike.roi.indexTarget(lon,lat)

            # ADW: Should allow fit_distance to float in order to model search procedure?
            # Currently we are evaluating at the true distance.
            #fit_distance = float(distance_modulus)
            distance_idx = np.fabs(grid.distance_modulus_array-distance_modulus).argmin()
            fit_distance = grid.distance_modulus_array[distance_idx]
            grid.search(coords=(lon,lat),distance_modulus=fit_distance)

            logger.info(str(self.loglike))

            mle = grid.mle()
            #results[i]['kernel'] = simulator.kernel.name
            results[i]['FIT_KERNEL'] = grid.loglike.kernel.name
            results[i]['TS'] = 2*grid.log_likelihood_sparse_array[distance_idx][pix]
            results[i]['FIT_TS'] = 2*np.max(grid.log_likelihood_sparse_array[:,pix])
            results[i]['FIT_MASS'] = grid.stellar_mass_conversion*mle['richness']
            results[i]['FIT_DISTANCE'] = fit_distance #mle['distance_modulus']

            err = grid.err()
            richness_err = (err['richness'][1]-err['richness'][0])/2.
            results[i]['FIT_MASS_ERR'] = grid.stellar_mass_conversion*richness_err

            distance_modulus_err = (err['distance_modulus'][1]-err['distance_modulus'][0])/2.
            results[i]['FIT_DISTANCE_ERR'] = distance_modulus_err

            logger.info("Fit parameter values:")
            for d in dtype:
                logger.info('\t%s: %s'%(d[0], results[i][d[0]]))

            if (i%self.config['simulate']['save'])==0 and outfile:
                ugali.utils.fileio.write(outfile,results)
                
        if outfile: ugali.utils.fileio.write(outfile,results,clobber=True)
            
        return results

if __name__ == "__main__":
    import ugali.utils.parser
    parser = ugali.utils.parser.Parser(description=__doc__)
    parser.add_config()
    parser.add_argument('-p','--popfile',default=None,
                        help='simulated population input file')
    parser.add_argument('-c','--catfile',default=None,
                        help='simulated catalog input file')
    parser.add_argument('-o','--outfile',default=None,
                        help='output results file')
    parser.add_argument('-i','--mc-source-id',default=None,type=int,
                        help='output results file')
    parser.add_debug()
    parser.add_verbose()
    args = parser.parse_args()

    analyzer = Analyzer(args.config,args.catfile,args.popfile)
    analyzer.run(outfile=args.outfile,mc_source_id=args.mc_source_id)
