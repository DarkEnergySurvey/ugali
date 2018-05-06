#!/usr/bin/env python
"""
Download Padova isochrones from:
http://stev.oapd.inaf.it/cgi-bin/cmd

Adapted from ezpadova by Morgan Fouesneau:
https://github.com/mfouesneau/ezpadova

"""
import os
try:
    from urllib.parse import urlencode
    from urllib.request import urlopen
except ImportError:
    from urllib import urlencode
    from urllib2 import urlopen
import re
import subprocess
from multiprocessing import Pool
from collections import OrderedDict as odict
import copy

import numpy as np
from ugali.utils.logger import logger
from ugali.utils.shell import mkdir
from ugali.analysis.isochrone import PadovaIsochrone

# survey system
photsys_dict = odict([
        ('des' ,'tab_mag_odfnew/tab_mag_decam.dat'),
        ('sdss','tab_mag_odfnew/tab_mag_sloan.dat'),
        ('ps1' ,'tab_mag_odfnew/tab_mag_panstarrs1.dat'),
])

photname_dict = odict([
        ('des' ,'DECAM'),
        ('sdss','SDSS'),
        ('ps1' ,'Pan-STARRS1'),
])

# Commented options may need to be restored for older version/isochrones.
# The parameters were tracked down by:
# Chrome -> View -> Developer -> Developer Tools
# Network -> Headers -> Request Payload

defaults_cmd=  {#'binary_frac': 0.3,
                #'binary_kind': 1,
                #'binary_mrinf': 0.7,
                #'binary_mrsup': 1,
                'cmd_version': 2.7,
                'dust_source': 'nodust',
                'dust_sourceC': 'nodustC',
                'dust_sourceM': 'nodustM',
                'eta_reimers': 0.2,
                #'extinction_av': 0,
                #'icm_lim': 4,
                'imf_file': 'tab_imf/imf_chabrier_lognormal.dat',
                'isoc_age': 1e9,
                'isoc_age0': 12.7e9,
                'isoc_dlage': 0.05,
                'isoc_dz': 0.0001,
                'isoc_kind': 'parsec_CAF09_v1.2S',
                'isoc_lage0': 6.602,   #Minimum allowed age                 
                'isoc_lage1': 10.1303, #Maximum allowed age                 
                'isoc_val': 0,               
                'isoc_z0': 0.0001,     #Minimum allowed metallicity         
                'isoc_z1': 0.03,       #Maximum allowed metallicity   
                'isoc_zeta': 0.0002,
                'isoc_zeta0': 0.0002,
                'kind_cspecmag': 'aringer09',
                'kind_dust': 0,
                'kind_interp': 1,
                'kind_mag': 2,
                'kind_postagb': -1,
                'kind_pulsecycle': 0,
                #'kind_tpagb': 0,
                #'lf_deltamag': 0.2,
                #'lf_maginf': 20,
                #'lf_magsup': -20,
                #'mag_lim': 26,
                #'mag_res': 0.1,
                'output_evstage': 1,
                'output_gzip': 0,
                'output_kind': 0,
                'photsys_file': photsys_dict['des'],
                #'photsys_version': 'yang',
                'submit_form': 'Submit'}

defaults_27 = dict(defaults_cmd,cmd_version='2.7')
defaults_28 = dict(defaults_cmd,cmd_version='2.8')
defaults_29 = dict(defaults_cmd,cmd_version='2.9')
defaults_30 = dict(defaults_cmd,cmd_version='3.0')


class Download(object):

    isochrone = None
    
    def __init__(self,survey='des',**kwargs):
        self.survey=survey.lower()

    def create_grid(self,abins,zbins):
        arange = np.linspace(abins[0],abins[1],abins[2]+1)
        zrange = np.logspace(np.log10(zbins[0]),np.log10(zbins[1]),zbins[2]+1)
        aa,zz = np.meshgrid(arange,zrange)
        return aa.flatten(),zz.flatten()

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

class Padova(Download):
    defaults = copy.deepcopy(defaults_27)
    isochrone = PadovaIsochrone
    
    abins = np.arange(1.0, 13.5 + 0.1, 0.1)
    zbins = np.arange(1e-4,1e-3 + 1e-5,1e-5)

    def query_server(self,outfile,age,metallicity):
        epsilon = 1e-4
        lage = np.log10(age*1e9)
        lage_min,lage_max = self.defaults['isoc_lage0'],self.defaults['isoc_lage1']
        if not (lage_min-epsilon < lage <lage_max+epsilon):
            msg = 'Age outside of valid range: %g [%g < log(age) < %g]'%(lage,lage_min,lage_max)
            raise RuntimeError(msg)

        z_min,z_max = self.defaults['isoc_z0'],self.defaults['isoc_z1']
        if not (z_min <= metallicity <= z_max):
            msg = 'Metallicity outside of valid range: %g [%g < z < %g]'%(metallicity,z_min,z_max)
            raise RuntimeError(msg)

        d = dict(self.defaults)
        d['photsys_file'] = photsys_dict[self.survey]
        d['isoc_age']     = age * 1e9
        d['isoc_zeta']    = metallicity

        server = 'http://stev.oapd.inaf.it'
        url = server + '/cgi-bin/cmd_%s'%d['cmd_version']
        logger.debug("Accessing %s..."%url)

        q = urlencode(d)
        logger.debug(url+'?'+q)
        c = urlopen(url, q).read()
        aa = re.compile('output\d+')
        fname = aa.findall(c)
        
        if len(fname) == 0:
            msg = "Output filename not found"
            raise RuntimeError(msg)

        out = '{0}/~lgirardi/tmp/{1}.dat'.format(server, fname[0])
        cmd = 'wget %s -O %s'%(out,outfile)
        logger.debug(cmd)
        stdout = subprocess.check_output(cmd,shell=True,stderr=subprocess.STDOUT)
        logger.debug(stdout)

        return outfile

    def verify(cls, filename, survey, age, metallicity):
        age = age*1e9
        nlines=14
        with open(filename,'r') as f:
            lines = [f.readline() for i in range(nlines)]
            if len(lines) < nlines:
                msg = "Incorrect file size"
                raise Exception(msg)
                
            try:
                s = lines[2].split()[-2]
                assert dict_output[survey][:4] in s
            except:
                msg = "Incorrect survey:\n"+lines[2]
                raise Exception(msg)

            try:
                z = lines[5].split()[2]
                assert np.allclose(metallicity,float(z),atol=1e-3)
            except:
                msg = "Metallicity does not match:\n"+lines[5]
                raise Exception(msg)

            try:
                a = lines[13].split()[1]
                assert np.allclose(age,float(a),atol=1e-5)
            except:
                msg = "Age does not match:\n"+lines[13]
                raise Exception(msg)


    @classmethod
    def verify(cls, filename, survey, age, metallicity):
        age = age*1e9
        nlines=15
        with open(filename,'r') as f:
            lines = [f.readline() for i in range(nlines)]
            if len(lines) < nlines:
                msg = "Incorrect file size"
                raise Exception(msg)

            for i,l in enumerate(lines):
                if l.startswith('# Photometric system:'): break
            else:
                msg = "Incorrect file header"
                raise Exception(msg)

            try:
                s = lines[i].split()[3]
                assert photname_dict[survey] == s
            except:
                msg = "Incorrect survey:\n"+lines[i]
                raise Exception(msg)

            try:
                z = lines[-1].split()[0]
                assert np.allclose(metallicity,float(z),atol=1e-5)
            except:
                msg = "Metallicity does not match:\n"+lines[-1]
                raise Exception(msg)

            try:
                a = lines[-1].split()[1]
                # Need to deal with age or log-age
                assert (np.allclose(age,float(a),atol=1e-2) or
                        np.allclose(np.log10(age),float(a),atol=1e-2))
            except:
                msg = "Age does not match:\n"+lines[-1]
                raise Exception(msg)

class Girardi2002(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'gi2000'

class Marigo2008(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'ma08'

class Girardi2010a(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'gi10a'

class Girardi2010b(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'gi10b'

class Bressan2012(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'parsec_CAF09_v1.2S'

class Marigo2017(Padova):
    defaults = dict(defaults_30)
    defaults['isoc_kind'] = 'parsec_CAF09_v1.2S_NOV13'

def factory(name, **kwargs):
    from ugali.utils.factory import factory
    return factory(name, module=__name__, **kwargs)

if __name__ == "__main__":
    import ugali.utils.parser
    description = "Download isochrones"
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_verbose()
    parser.add_force()
    parser.add_argument('-a','--age',default=None,type=float)
    parser.add_argument('-z','--metallicity',default=None,type=float)
    parser.add_argument('-k','--kind',default='Bressan2012')
    parser.add_argument('-s','--survey',default='des')
    parser.add_argument('-o','--outdir',default=None)
    parser.add_argument('-n','--njobs',default=10,type=int)
    args = parser.parse_args()

    if args.verbose:
        try:                                                                                            
            from http.client import HTTPConnection
        except ImportError:                                                                     from httplib import HTTPConnection
        HTTPConnection.debuglevel = 1

    if args.outdir is None: 
        args.outdir = os.path.join(args.survey.lower(),args.kind.lower())
    logger.info("Writing to output directory: %s"%args.outdir)

    p = factory(args.kind,survey=args.survey)

    # Defaults
    abins = [args.age] if args.age else p.abins
    zbins = [args.metallicity] if args.metallicity else p.zbins
    grid = [g.flatten() for g in np.meshgrid(abins,zbins)]
    logger.info("Ages:\n  %s"%np.unique(grid[0]))
    logger.info("Metallicities:\n  %s"%np.unique(grid[1]))

    def run(args):
        try:  
            p.download(*args)
        except Exception as e: 
            logger.warn(str(e))
            logger.error("Download failed.")

    arguments = [(a,z,args.outdir,args.force) for a,z in zip(*grid)]
    if args.njobs > 1:
        pool = Pool(processes=args.njobs, maxtasksperchild=100)
        results = pool.map(run,arguments)
    else:
        results = list(map(run,arguments))
    
