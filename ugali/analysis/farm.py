#!/usr/bin/env python
"""
Dispatch the likelihood scan to a cluster.
"""

import os
from os.path import join
import shutil
import time

import numpy as np
import healpy as hp

import ugali.utils.config
import ugali.utils.skymap
import ugali.utils.batch

from ugali.utils.projector import gal2cel
from ugali.utils.healpix import query_disc
from ugali.utils.healpix import pix2ang,ang2vec
from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

class Farm:
    """ Class for organizing and submitting likelihood scan jobs. """

    def __init__(self, configfile, verbose=False):
        self.configfile = configfile
        self.config = ugali.utils.config.Config(configfile)
        self._setup()
        self.verbose = verbose

    def _setup(self):
        self.nside_catalog    = self.config['coords']['nside_catalog']
        self.nside_likelihood = self.config['coords']['nside_likelihood']
        self.nside_pixel      = self.config['coords']['nside_pixel']

        self.filenames = self.config.getFilenames()
        self.skip = "Outfile already exists. Skipping..."

        # Might consider storing only the good filenames
        # self.filenames = self.filenames.compress(~self.filenames.mask['pix'])
        self.catalog_pixels = self.filenames['pix'].compressed()

    def command(self, outfile, configfile, pix):
        """
        Generate the command for running the likelihood scan.
        """
        params = dict(script=self.config['scan']['script'],
                      config=configfile, outfile=outfile, 
                      nside=self.nside_likelihood, pix=pix,
                      verbose='-v' if self.verbose else '')
        cmd = '%(script)s %(config)s %(outfile)s --hpx %(nside)i %(pix)i %(verbose)s'%params
        return cmd
        
    def submit_all(self, coords=None, queue=None, debug=False):
        """
        Submit likelihood analyses on a set of coordinates. If
        coords is `None`, submit all coordinates in the footprint.

        Inputs:
        coords : Array of target locations in Galactic coordinates. 
        queue  : Overwrite submit queue.
        debug  : Don't run.
        """
        if coords is None:
            pixels = np.arange(hp.nside2npix(self.nside_likelihood))
        else:
            lon,lat,radius = coords['lon'],coords['lat'],coords['radius']

            #ADW: coords are always parsed in GAL, so convert to CEL if necessary
            if self.config['coords']['coordsys'].lower() == 'cel':
                lon,lat = gal2cel(lon,lat)

            vec = ang2vec(lon,lat)
            pixels = np.zeros(0, dtype=int)
            for v,r in zip(vec,radius):
                pix = query_disc(self.nside_likelihood,v,r,inclusive=True,fact=32)
                pixels = np.hstack([pixels, pix])
            #pixels = np.unique(pixels)

        inside = ugali.utils.skymap.inFootprint(self.config,pixels)
        if inside.sum() != len(pixels):
            logger.warning("Ignoring pixels outside survey footprint:\n"+str(pixels[~inside]))
        if inside.sum() == 0:
            logger.warning("No pixels inside footprint.")
            return

        # Only write the configfile once
        outdir = mkdir(self.config['output']['likedir'])
        # Actually copy config instead of re-writing
        shutil.copy(self.config.filename,outdir)
        configfile = join(outdir,os.path.basename(self.config.filename))

        pixels = pixels[inside]
        self.submit(pixels,queue=queue,debug=debug,configfile=configfile)

    def submit(self, pixels, queue=None, debug=False, configfile=None):
        """
        Submit the likelihood job for the given pixel(s).
        """
        # For backwards compatibility
        batch = self.config['scan'].get('batch',self.config['batch'])
        queue = batch.get('default','medium') if queue is None else queue

        # Need to develop some way to take command line arguments...
        self.batch = ugali.utils.batch.batchFactory(queue,**batch.get(queue,{}))
        self.batch.max_jobs = self.config['scan'].get('max_jobs',200)

        if np.isscalar(pixels): pixels = np.array([pixels])

        outdir = mkdir(self.config['output']['likedir'])
        logdir = mkdir(join(outdir,'log'))
        subdir = mkdir(join(outdir,'sub'))

        # Save the current configuation settings; avoid writing 
        # file multiple times if configfile passed as argument.
        if configfile is None:
            shutil.copy(self.config.filename,outdir)
            configfile = join(outdir,os.path.basename(self.config.filename))
                
        lon,lat = pix2ang(self.nside_likelihood,pixels)
        commands = []
        chunk = self.config['scan'].get('chunk',25)
        istart = 0
        logger.info('=== Submit Likelihood ===')
        for ii,pix in enumerate(pixels):
            msg = '  (%i/%i) pixel=%i nside=%i; (lon, lat) = (%.2f, %.2f)'
            msg = msg%(ii,len(pixels),pix, self.nside_likelihood,lon[ii],lat[ii])
            logger.info(msg)

            # Create outfile name
            outfile = self.config.likefile%(pix,self.config['coords']['coordsys'].lower())
            outbase = os.path.basename(outfile)
            jobname = batch.get('jobname','ugali')

            # Submission command
            sub = not os.path.exists(outfile)
            cmd = self.command(outfile,configfile,pix)
            commands.append([ii,cmd,lon[ii],lat[ii],sub])
            
            if chunk == 0:
                # No chunking
                command = cmd
                submit = sub
                logfile = join(logdir,os.path.splitext(outbase)[0]+'.log')
            elif (len(commands)%chunk==0) or (ii+1 == len(pixels)):
                # End of chunk, create submission script
                commands = np.array(commands,dtype=object)
                istart, iend = commands[0][0], commands[-1][0]
                subfile = join(subdir,'submit_%08i_%08i.sh'%(istart,iend))
                logfile = join(logdir,'submit_%08i_%08i.log'%(istart,iend))
                command = "sh %s"%subfile

                submit = np.any(commands[:,-1])
                if submit: self.write_script(subfile,commands)
            else:
                # Not end of chunk
                continue
            commands=[]

            # Actual job submission
            if not submit:
                logger.info(self.skip)
                continue
            else:
                job = self.batch.submit(command,jobname,logfile)
                logger.info("  "+job)
                time.sleep(0.5)

    def write_script(self, filename, commands):
        """ Write a batch submission script.

        Parameters
        ----------
        filename : filename of batch script
        commands : list of commands to execute

        Returns
        -------
        None
        """
        info = 'echo "{0:=^60}";\n'
        hline = info.format("")
        newline = 'echo;\n'
        shebang = "#!/usr/bin/env bash"
        # Limit the memory based on SLAC 4 GB per node (defined in KB)
        # Careful, shell arithmetic is weird.
        memory_limit = """
if [ -n "$LSB_CG_MEMLIMIT" ] & [ -n "$LSB_HOSTS" ]; then
    mlimit=$(( $(wc -w <<< $LSB_HOSTS) * $LSB_CG_MEMLIMIT/1024 * 9/10 ))
    ulimit -v ${mlimit}; ulimit -H -v ${mlimit};
fi
"""
        memory_usage=r"""free -m | awk 'NR==2{printf "Memory Usage: %.2f/%.2fGB (%.2f%%)\n",$3/1024,$2/1024,$3*100/$2}';"""
        memory_usage=r"""ps -U $USER --no-headers -o rss | awk '{sum+=$1} END {print "Memory Usage: " int(sum/1024**2) "GB"}'"""

        istart, iend = commands[0][0], commands[-1][0]
        script = open(filename,'w')
        script.write(shebang)
        #script.write(memory_limit)
        script.write(hline)
        script.write(info.format('Submit Jobs %i to %i'%(istart,iend)))
        script.write(hline)
        script.write(newline)
        script.write('status=0;\n')
        for i,cmd,lon,lat,sub in commands: 
            script.write(info.format('Job %i: (%.2f, %.2f)'%(i,lon,lat)))
            if sub: 
                script.write(memory_usage+'\n')
                script.write('%s; [ $? -ne 0 ] && status=1;\n'%cmd)
            else:   
                script.write('echo "%s";\n'%self.skip)
            script.write(hline)
            script.write(newline)
        script.write('exit $status;\n')
        script.close()

if __name__ == "__main__":
    import ugali.utils.parser
    parser = ugali.utils.parser.Parser(description=__doc__)
    parser.add_config()
    parser.add_debug()
    parser.add_queue()
    parser.add_verbose()
    parser.add_coords(required=True,radius=True,targets=True)
    opts = parser.parse_args()

    farm = Farm(opts.config,verbose=opts.verbose)
    x = farm.submit_all(coords=opts.coords,queue=opts.queue,debug=opts.debug)
