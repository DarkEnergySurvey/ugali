#!/usr/bin/env python
"""
Analyze a simulated data set.
"""
__author__ = "Alex Drlica-Wagner"
import copy
import os
import time
import resource, psutil
from collections import OrderedDict as odict

import numpy
import numpy as np
import scipy.interpolate
import healpy
import numpy.lib.recfunctions as recfuncs
import fitsio

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
from ugali.utils import mlab

# Analysis flags
FLAGS = odict([])
FLAGS['FLAG_PROC'  ] = FLAG_PROC   = 0  # Simulation was processed
FLAGS['FLAG_NOPROC'] = FLAG_NOPROC = 1  # No processing
FLAGS['FLAG_NOBJ'  ] = FLAG_NOBJ   = 2  # Too many catalog objects
FLAGS['FLAG_FIT'   ] = FLAG_FIT    = 4  # Fit failure
FLAGS['FLAG_EBV'   ] = FLAG_EBV    = 8  # EBV value too large
FLAGS['FLAG_MEM'   ] = FLAG_MEM    = 16 # Memory error

def update_header_flags(filename):
    fits = fitsio.FITS(filename,'rw')
    for k,v in FLAGS.items():
        fits[1].write_key(k,v)

# Results dtypes
# FIXME: KERNEL should be removed for next run
DTYPES=[('TS','>f4'),
        ('FIT_KERNEL','S18'),('FIT_EXTENSION','>f4'),
        ('FIT_MASS','>f4'),('FIT_MASS_ERR','>f4'),
        ('FIT_DISTANCE','>f4'),('FIT_DISTANCE_ERR','>f4'),('FLAG','>i4'),
        ('RUNTIME','>f4'),('MEMORY','>i8')]

KB = 1024**1
MB = 1024**2
GB = 1024**3

class Analyzer(object):
    """
    Class for analyzing simulated data
    """
    def __init__(self, config, catfile=None, popfile=None):
        self.config = Config(config)
        self.population = self.read_population(popfile)
        self.catalog = self.read_catalog(catfile)
        self.mlimit = -1

    def get_memory_usage(self):
        """Get the memory usage of this process.

        Parameters
        ----------
        """
        process = psutil.Process()
        mem = process.memory_info()[0]
        return mem
    
        ## https://stackoverflow.com/a/7669482/4075339
        ## peak memory usage (kilobytes on Linux)
        #usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        #return usage
        
    def get_memory_limit(self):
        """Get the hard memory limit from LSF.

        Parameters
        ----------
        None
        
        Returns
        -------
        mlimit : memory limit (bytes)
        """
        rsrc = resource.RLIMIT_AS
        soft, hard = resource.getrlimit(rsrc)
        if os.getenv('LSB_CG_MEMLIMIT') and os.getenv('LSB_HOSTS'):
            # Memory limit per core
            memlimit = int(os.getenv('LSB_CG_MEMLIMIT'), 16)
            # Number of cores
            ncores = len(os.getenv('LSB_HOSTS').split())
            #soft = ncores * memlimit - 100*1024**2
            soft = ncores * memlimit - GB
        return soft

    def set_memory_limit(self, mlimit):
        """Set the (soft) memory limit through setrlimit.
        
        Parameters
        ----------
        mlimit : soft memory limit (bytes)
        
        Returns
        -------
        soft, hard : memory limits (bytes)
        """
        rsrc = resource.RLIMIT_AS
        resource.setrlimit(rsrc, (mlimit, mlimit))
        self.mlimit = mlimit
        return resource.getrlimit(rsrc)

        #rsrc = resource.RLIMIT_AS
        #soft, hard = resource.getrlimit(rsrc)
        #resource.setrlimit(rsrc, (mlimit, hard))
        #self.mlimit, hard = resource.getrlimit(rsrc)
        #return (self.mlimit,hard)


    def read_population(self, filename=None):
        if not filename:
            filename = os.path.join(self.config['simulate']['dirname'],self.config['simulate']['popfile'])
        logger.info("Reading population file: %s"%filename)
        pop = ugali.utils.fileio.read(filename)
        pop.dtype.names = list(map(str.upper,pop.dtype.names))
        return pop

    def read_catalog(self, filename=None):
        if not filename:
            filename = os.path.join(self.config['simulate']['dirname'],self.config['simulate']['catfile'])
        logger.info("Reading catalog file: %s"%filename)
        catalog =  ugali.observation.catalog.Catalog(self.config,filenames=filename)
        catalog.data = mlab.rec_append_fields(catalog.data,
                                           names=['PIX8','PIX4096'],
                                           arrs=np.zeros((2,len(catalog.lon)),dtype='>i8'))
        return catalog

    def read_results(self, filename):
        logger.info("Reading results file: %s"%filename)
        results = ugali.utils.fileio.read(filename)
        return results
        
    def create_results(self, population=None):
        """ Create the results array of len(population)

        Parameters:
        -----------
        population : population array (self.population if None)
        
        Returns:
        --------
        results : array of len(population)
        """
        if population is None: population = self.population
        size = len(population)
        results = np.array(np.nan*np.ones(size),dtype=DTYPES)
        results = recfuncs.merge_arrays([population,results],
                                             flatten=True,
                                             asrecarray=False,usemask=False)
        results['TS'] = -np.inf
        results['FLAG'] = FLAG_NOPROC
        results['MEMORY'] = -1

        return results

    def write_results(self, filename,**kwargs):
        """ Write results array to a file.

        Parameters:
        -----------
        filename : output file name
        kwargs   : arguments passed to fileio.write
        
        Returns:
        --------
        None
        """
        ugali.utils.fileio.write(filename,self.results,**kwargs)
        update_header_flags(filename)

    def runall(self,outfile=None,mc_source_id=None,rerun=False):
        """Run all sources in population.
        
        Parameters:
        -----------
        outfile      : file to write output to
        mc_source_id : list of sources to process (None is all sources)

        Returns:
        --------
        results      : processing results
        """
        if mc_source_id is None:
            mc_source_id = np.unique(self.catalog.mc_source_id)

        # Select only systems that are in the catalog
        sel = np.in1d(self.population['MC_SOURCE_ID'],mc_source_id)
        
        if not sel.sum():
            msg = "Requested MC_SOURCE_IDs not found in population."
            raise ValueError(msg)

        if os.path.exists(outfile) and rerun:
            # read the results from the existing outfile
            self.results = self.read_results(outfile)
        else:
            # create the results
            self.results = self.create_results(population=self.population[sel])
        
        if not np.in1d(mc_source_id,self.results['MC_SOURCE_ID']).all():
            msg = "Requested MC_SOURCE_IDs not found in results."
            raise ValueError(msg)
    
        if outfile: 
            logger.info("Writing %s..."%outfile)
            self.write_results(outfile,clobber=True)

        for i,r in enumerate(self.results):
            # Skip if not in mc_source_id list
            if self.results[i]['MC_SOURCE_ID'] not in mc_source_id:
                msg = "%i skipped."%self.results[i]['MC_SOURCE_ID']
                logger.info(msg)
                continue

            # Rerun in NOPROC or MEM flagged
            if (self.results[i]['FLAG'] & (FLAG_NOPROC | FLAG_MEM)) == 0:
                msg = "%i already processed."%self.results[i]['MC_SOURCE_ID']
                logger.info(msg)
                continue

            start_time = time.time()

            try: 
                self.runone(i)
            except MemoryError as e:
                msg = "Memory usage exceeded %.3f GB"%(self.mlimit/GB)
                logger.warn(msg)
                self.results[i]['FLAG'] |= FLAG_MEM
            except Exception as e:
                logger.error(str(e))
                self.results[i]['FLAG'] |= FLAG_FIT
                
            runtime = time.time() - start_time
            self.results[i]['MEMORY'] = self.get_memory_usage()
            self.results[i]['RUNTIME'] = runtime

            logger.info("Fit parameter values:")
            for d in DTYPES:
                logger.info('\t%s: %s'%(d[0], self.results[i][d[0]]))

            logger.info("Memory usage: %.3f GB"%(self.get_memory_usage()/GB))

            if (i%self.config['simulate']['save'])==0 and outfile:
                logger.info("Writing %s..."%outfile)
                self.write_results(outfile,clobber=True)

        if outfile: 
            logger.info("Writing %s..."%outfile)
            self.write_results(outfile,clobber=True)

        return self.results

    #from memory_profiler import profile
    #@profile
    def runone(self, i):
        """ Run one simulation.

        Parameters:
        -----------
        i : index of the simulation to run
        
        Returns:
        --------
        results : result array
        """
        results = self.results
        results[i]['FLAG'] = FLAG_PROC

        params = dict(list(zip(results[i].dtype.names,results[i])))
        size = len(results)
        lon,lat = params['RA'],params['DEC']
        distance_modulus = params['DISTANCE_MODULUS']
        mc_source_id = params['MC_SOURCE_ID']
        extension=np.degrees(np.arctan(params['R_PHYSICAL']/params['DISTANCE']))

        logger.info('\n(%i/%i); (id, lon, lat, mod, ext) = (%i, %.2f, %.2f, %.1f, %.3f)'%(i+1,size,mc_source_id,lon,lat,distance_modulus,extension))
        
        if params['EBV'] > 0.2:
            msg = "High reddening region; skipping..."
            logger.warn(msg)
            results[i]['FLAG'] |= FLAG_EBV
            #raise Exception(msg)
            #results[i]['TS'] = np.nan
            #return

        # This links to the parameters in the data scan
        section = 'scan'
        # This links to the parameters in the simulate section
        #section = 'simulate'

        source=ugali.analysis.loglike.createSource(self.config,section=section,lon=lon,lat=lat)
        
        logger.info("Reading data catalog...")
        obs = ugali.analysis.loglike.createObservation(self.config,lon=lon,lat=lat)
            

        # Select just the simulated target of interest
        logger.info("Merging simulated catalog...")
        data = self.catalog.data[self.catalog.mc_source_id == mc_source_id].copy()
        data = np.array(data[list(obs.catalog.data.dtype.names)],
                        dtype=obs.catalog.data.dtype)
        obs.catalog = ugali.observation.catalog.Catalog(self.config, data=np.concatenate([obs.catalog.data,data]))

        loglike = ugali.analysis.loglike.LogLikelihood(self.config,obs,source)

        # Mitigate memory overflow issues by cutting objects with
        # too many catalog stars
        if len(loglike.catalog) > 5e5: # 1e5
            msg = "Large catalog (N_CATALOG = %i)."%len(loglike.catalog)
            logger.warn(msg)
            results[i]['FLAG'] |= FLAG_NOBJ

        grid = ugali.analysis.scan.GridSearch(self.config,loglike)
        self.grid = grid
        self.loglike = self.grid.loglike

        pix = self.loglike.roi.indexTarget(lon,lat)

        # ADW: Should fit_distance be free in order to model search procedure?
        if self.config['simulate'].get('fit_distance',False):
            fit_distance = None
        else:
            idx = np.fabs(grid.distance_modulus_array-distance_modulus).argmin()
            fit_distance = grid.distance_modulus_array[idx]

        try:
            grid.search(coords=(lon,lat),distance_modulus=fit_distance)
            results[i]['FLAG'] &= ~FLAG_FIT
        except ValueError as e:
            logger.error(str(e))
            results[i]['FLAG'] |= FLAG_FIT

        mle = grid.mle()
        
        distance_idx = np.fabs(grid.distance_modulus_array-mle['distance_modulus']).argmin()

        ts = 2*grid.loglike_array[distance_idx][pix]

        results[i]['FIT_KERNEL'] = grid.loglike.kernel.name
        results[i]['TS'] = ts
        results[i]['FIT_MASS'] = grid.stellar_mass_conversion*mle['richness']
        results[i]['FIT_DISTANCE'] = mle['distance_modulus']
        results[i]['FIT_EXTENSION'] = grid.loglike.kernel.extension
        
        err = grid.err()
        richness_err = (err['richness'][1]-err['richness'][0])/2.
        results[i]['FIT_MASS_ERR'] = grid.stellar_mass_conversion*richness_err
             
        distance_modulus_err = (err['distance_modulus'][1]-err['distance_modulus'][0])/2.
        results[i]['FIT_DISTANCE_ERR'] = distance_modulus_err

        """
        fit_extension = [0.02, 0.07, 0.15] # half light radii (deg)

        # ADW: This won't work since we add extension inside the search loop
        for ext in fit_extension:
            grid.loglike.set_params(extension = ext)
            # Fit failures are often due to fracdet = 0
            try:
                grid.search(coords=(lon,lat),distance_modulus=fit_distance)
                results[i]['FLAG'] &= ~FLAG_FIT
            except ValueError as e:
                logger.error(str(e))
                results[i]['FLAG'] |= FLAG_FIT
                continue
            mle = grid.mle()
            
            ts = 2*grid.loglike_array[distance_idx][pix]
            if ts <= results[i]['TS']: 
                logger.info("No TS increase; continuing...")
                continue

            results[i]['FIT_KERNEL'] = grid.loglike.kernel.name
            results[i]['TS'] = ts
            results[i]['FIT_MASS'] = grid.stellar_mass_conversion*mle['richness']
            results[i]['FIT_DISTANCE'] = fit_distance #mle['distance_modulus']
            results[i]['FIT_EXTENSION'] = ext

            err = grid.err()
            richness_err = (err['richness'][1]-err['richness'][0])/2.
            results[i]['FIT_MASS_ERR'] = grid.stellar_mass_conversion*richness_err
             
            distance_modulus_err = (err['distance_modulus'][1]-err['distance_modulus'][0])/2.
            results[i]['FIT_DISTANCE_ERR'] = distance_modulus_err
            """
        return results

    run = runall

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
    parser.add_argument('-i','--mc-source-id',default=None,type=int,action='append',
                        help='specific source id to run')
    parser.add_argument('-m','--mlimit',default=None,type=int,
                        help='limit memory usage')
    parser.add_argument('-r','--rerun',action='store_true',
                        help='rerun failed jobs')

    #parser.add_force()
    #parser.add_debug()
    parser.add_verbose()
    args = parser.parse_args()

    analyzer = Analyzer(args.config,args.catfile,args.popfile)

    if args.mlimit is not None:
        if args.mlimit == 0:
            mlimit = analyzer.get_memory_limit()
        else:
            mlimit = args.mlimit * GB
        soft,hard = analyzer.set_memory_limit(mlimit)
        logger.info("Setting memory limit to %.3f GB"%(soft/GB))

    if args.mc_source_id is None:
        basename = os.path.splitext(args.catfile)[0]
        imin,imax = list(map(int,basename.rsplit('_',1)[-1].split('-')))
        args.mc_source_id = np.arange(imin,imax+1)

    analyzer.run(outfile=args.outfile,mc_source_id=args.mc_source_id,
                 rerun=args.rerun)
