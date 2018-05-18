#!/usr/bin/env python

import os, sys
import re
import io
import subprocess

import numpy as np

try:
    import http.client as httpcl
except ImportError:
    import httplib as httpcl
 
from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

DATABASES = {
    'sdss':['dr10'],
    'des' :['sva1','sva1_gold','y1a1','y2n','y2q1'],
    }

def databaseFactory(survey,release):
    if survey == 'sdss':
        return SDSSDatabase(release = release)
    elif survey == 'des':
        if release == 'y1a1':
            return Y1A1Database()
        elif release == 'y2n':
            return Y2NDatabase()
        elif release == 'y2q1':
            return Y2Q1Database()
        else:
            msg = "Unrecognized release: %s"%release
            raise Exception(msg)
        return DESDatabase(release = release)
    else:
        logger.error("Unrecognized survey: %s"%survey)
        return None

class Database(object):
    def __init__(self):
        pass

    def load_pixels(self, pixfile=None):
        if pixfile is not None:
            self.pixels = np.loadtxt(pixfile,dtype=[('name',int),('ra_min',float),('ra_max',float),
                                                    ('dec_min',float),('dec_max',float)])
            # One-line input file
            if self.pixels.ndim == 0: self.pixels = np.array([self.pixels])
        else:
            self.pixels = self.create_pixels()

    @staticmethod
    def create_pixels(nra=18, ndec=10):
            ra_step = 360./nra #20
            ra_range = np.around(np.arange(0,360+ra_step,ra_step),0)
            # Decreasing dec...
            sin_dec_step = -(2.0/ndec) # -0.2
            dec_range = np.around(np.degrees(np.arcsin(np.arange(1,-1+sin_dec_step,sin_dec_step))),0)

            xx, yy = np.meshgrid( ra_range,dec_range)
            ra_min  = xx[1:,:-1].flatten(); ra_max = xx[1:,1:].flatten()
            dec_min = yy[1:,1:].flatten(); dec_max = yy[:-1,1:].flatten() # Decreasing...
            name = np.arange(len(ra_min),dtype=int)
            return  np.rec.fromarrays([name, ra_min, ra_max, dec_min, dec_max],
                                      dtype=[('name',int),('ra_min',float),('ra_max',float),
                                             ('dec_min',float),('dec_max',float)])


    def generate_query(self):
        """ Should be implemented by child class. """
        pass

    def download(self):
        """ Should be implemented by child class. """        
        pass

    def run(self):
        """ Should be implemented by child class. """
        pass

class SDSSDatabase(Database):
    """
    For downloading SDSS DR10 data set.
    """
    def __init__(self,release='DR10'):
        super(SDSSDatabase,self).__init__()
        self.release = release.lower()
        self.basename = "sdss_%s_photometry"%self.release

    def _setup_casjobs(self):
        # Function here to install casjobs.jar and CasJobs.config...
        # "wget http://skyserver.sdss3.org/CasJobs/download/casjobs.jar"
        # "wget http://skyserver.sdss3.org/CasJobs/download/CasJobs.config.x -o CasJobs.config"

        # For now, just check that they exist
        files = ['casjobs.jar','CasJobs.config']
        for f in files:
            if not os.path.exists(f):
                msg = "Can't find: %s"%f
                raise IOError(msg)

    def generate_query(self, ra_min,ra_max,dec_min,dec_max,filename,db):
        outfile = open(filename,"w")
        outfile.write('SELECT s.objID, s.ra AS "RA", s.dec as "DEC",\n')
        outfile.write('s.psfmag_g AS "MAG_PSF_G",\n')
        outfile.write('s.psfmagerr_g AS "MAGERR_PSF_G",\n')
        outfile.write('s.psfmag_g - s.extinction_g AS "MAG_PSF_SFD_G",\n')
        outfile.write('s.psfmag_r AS "MAG_PSF_R",\n')
        outfile.write('s.psfmagerr_r AS "MAGERR_PSF_R",\n')
        outfile.write('s.psfmag_r - s.extinction_r AS "MAG_PSF_SFD_R",\n')
        outfile.write('s.psfmag_i AS "MAG_PSF_I",\n')
        outfile.write('s.psfmagerr_i AS "MAGERR_PSF_I",\n')
        outfile.write('s.psfmag_i - s.extinction_i AS "MAG_PSF_SFD_I",\n')
        outfile.write('s.psfmag_z AS "MAG_PSF_Z",\n')
        outfile.write('s.psfmagerr_z AS "MAGERR_PSF_Z",\n')
        outfile.write('s.psfmag_z - s.extinction_z AS "MAG_PSF_SFD_Z"\n')
        outfile.write('INTO MyDB.%s\n' % (db))
        outfile.write('FROM %s.StarTag as s\n'%(self.release))
        outfile.write('WHERE s.ra > %.7f AND s.ra < %.7f\n' % (ra_min,ra_max))
        outfile.write('AND s.dec > %.7f AND s.dec < %.7f\n' % (dec_min,dec_max))
        outfile.write('AND s.clean = 1\n')
        outfile.close()

    def query(self,dbase,task,query):
        logger.info("Running query...")
        cmd = "java -jar casjobs.jar run -t %s -n %s -f %s" % (dbase,task,query)
        logger.info(cmd)
        ret = subprocess.check_output(cmd,shell=True,stderr=subprocess.STDOUT) 
        if 'ERROR:' in ret:
            raise subprocess.CalledProcessError(1,cmd,ret)
        return ret
        
    def extract(self, table):
        logger.info("Extracting...")
        cmd = "java -jar casjobs.jar extract -u -F -a FITS -b %s" % (table)
        logger.info(cmd)
        retval = subprocess.check_output(cmd,shell=True)

        url = None
        match = re.search("(http\:\/\/.*\.fit)",retval)
        if (match is not None) :
            url = match.group(0)
        else:
            logger.info("URL not found...here's the output")
            logger.info(retval)
        return url

    def wget(self,url,outfile=None):
        logger.info("Downloading %s\n" % (url))
        if outfile is not None: cmd = "wget -O %s %s" % (outfile,url)
        else:                   cmd = "wget %s" % (url)
        logger.info(cmd)
        return subprocess.check_output(cmd,shell=True)

    def drop(self, table):
        logger.info("Dropping...")
        cmd = "java -jar casjobs.jar execute -t MyDB -n \"drop query\" \"drop table %s\""%(table)
        logger.info(cmd)
        return subprocess.check_output(cmd,shell=True)

    def download(self, pixel, outdir=None, force=False):
        if outdir is None: outdir = './'
        else:              mkdir(outdir)
        sqldir = mkdir(os.path.join(outdir,'sql'))
        self._setup_casjobs()

        basename = self.basename + "_%04d"%pixel['name']
        sqlname = os.path.join(sqldir,basename+'.sql')
        dbname = basename+'_output'
        taskname = basename
        outfile = os.path.join(outdir,basename+".fits")
        if os.path.exists(outfile) and not force:
            logger.warning("Found %s; skipping..."%(outfile))
            return

        logger.info("\nDownloading pixel: %(name)i (ra=%(ra_min)g:%(ra_max)g,dec=%(dec_min)g:%(dec_max)g)"%(pixel))
        logger.info("Working on "+sqlname)
         
        self.generate_query(pixel['ra_min'],pixel['ra_max'],pixel['dec_min'],pixel['dec_max'],sqlname,dbname)

        try:
            self.query(self.release,taskname,sqlname)
        except subprocess.CalledProcessError as e:
            logger.error(e.output)
            self.drop(dbname)
            raise e
        
        try:
            url = self.extract(dbname)
        except subprocess.CalledProcessError as e:
            self.drop(dbname)
            raise e
            
        if (url is not None):
            self.wget(url,outfile)

        self.drop(dbname)

    def upload(self, array, fields=None, table="MyDB", configfile=None):
        """
        Upload an array to a personal database using SOAP POST protocol.
        http://skyserver.sdss3.org/casjobs/services/jobs.asmx?op=UploadData
        """

        wsid=''
        password=''
        if configfile is None:
            configfile = "CasJobs.config"
        logger.info("Reading config file: %s"%configfile)
        lines = open(configfile,'r').readlines()
        for line in lines:
            k,v = line.strip().split('=')
            if k == 'wsid': wsid = v
            if k == 'password': password = v

        logger.info("Attempting to drop table: %s"%table)
        self.drop(table)
     
        SOAP_TEMPLATE = """
        <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
                         xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
                         xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
          <soap12:Body>
            <UploadData xmlns="http://Services.Cas.jhu.edu">
              <wsid>%s</wsid>
              <pw>%s</pw>
              <tableName>%s</tableName>
              <data>%s</data>
              <tableExists>%s</tableExists>
            </UploadData>
          </soap12:Body>
        </soap12:Envelope>
        """
     
        logger.info("Writing array...")
        s = io.StringIO()
        np.savetxt(s,array,delimiter=',',fmt="%.10g")
        tb_data = ''
        if fields is not None: 
            tb_data += ','.join(f for f in fields)+'\n'
        tb_data += s.getvalue()
     
        message = SOAP_TEMPLATE % (wsid, password, table, tb_data, "false")
        
        #construct and send the header
        webservice = httpcl.HTTP("skyserver.sdss3.org")
        webservice.putrequest("POST", "/casjobs/services/jobs.asmx")
        webservice.putheader("Host", "skyserver.sdss3.org")
        webservice.putheader("Content-type", "text/xml; charset=\"UTF-8\"")
        webservice.putheader("Content-length", "%d" % len(message))
        webservice.endheaders()
        logger.info("Sending SOAP POST message...")
        webservice.send(message)
         
        # get the response
        statuscode, statusmessage, header = webservice.getreply()
        print("Response: ", statuscode, statusmessage)
        print("headers: ", header)
        res = webservice.getfile().read()
        print(res)


    def run(self,pixfile=None,outdir=None):
        self.load_pixels(pixfile)
        for pixel in self.pixels:
            self.download(pixel,outdir)

    def inFootprint(self, ra, dec):
        basename = self.basename + '_coverage'
        sqlname = basename + '.sql'
        dbname = basename + '_output'
        task = basename
        outfile = basename+".fits"

        # Upload the (ra,dec) coordinates to casjobs
        table = "ra_dec"
        self.upload(np.array([ra,dec]).T, ['ra','dec'],table=table)

        # Query the database for the footprint
        query = open(sqlname,'w')
        query.write('SELECT dbo.fInFootprintEq(t.ra, t.dec, 0)\n')
        query.write('INTO MyDB.%s\n' % (dbname))
        query.write('FROM MyDB.%s AS t'%table)
        query.close()

        self.query(self.release,task,sqlname)
        url = self.extract(dbname)
        
        if (url is not None):
            self.wget(url,outfile)

        self.drop(table)
        self.drop(dbname)


    def footprint(self,nside):
        """
        Download the survey footprint for HEALpix pixels.
        """
        import healpy
        import ugali.utils.projector
        if nside > 2**9: raise Exception("Overflow error: nside must be <=2**9")
        pix = np.arange(healpy.nside2npix(nside),dtype='int')
        footprint = np.zeros(healpy.nside2npix(nside),dtype='bool')
        ra,dec = ugali.utils.projector.pixToAng(nside,pix)
        table_name = 'Pix%i'%nside
        self.upload(np.array([pix,ra,dec]).T, ['pix','ra','dec'], name=table_name)
        radius = healpy.nside2resol(nside_superpix,arcmin=True)

        query="""
        SELECT t.pix, dbo.fInFootprintEq(t.ra, t.dec, %g)
        FROM %s AS t
        """%(radius, table_name)

class DESDatabase(Database):
    ####################################
    ###### !!! NOT IMPLEMENTED !!! #####
    ####################################

    def __init__(self,release='SVA1_GOLD'):
        super(DESDatabase,self).__init__()
        self.release = release.lower()
        self.basename = "des_%s_photometry"%self.release

    def _setup_desdbi(self):
        # Function here to setup trivialAccess client...
        # This should work but it doesn't
        import warnings
        warnings.warn("desdbi is deprecated", DeprecationWarning)
        import despydb.desdbi

    def generate_query(self, ra_min,ra_max,dec_min,dec_max,filename,db):
        # Preliminary and untested
        table = 'Y1A1_COADD_OBJECTS@DESSCI'

        outfile = open(filename,"w")
        outfile.write('SELECT s.COADD_OBJECTS_ID, s.RA, s.DEC, \n')
        outfile.write('s.MAG_PSF_G, s.MAGERR_PSF_G, \n')
        outfile.write('s.MAG_PSF_R, s.MAGERR_PSF_R, \n')
        outfile.write('s.MAG_PSF_I, s.MAGERR_PSF_I  \n')
        outfile.write('FROM %s s \n'%(table))
        outfile.write('WHERE s.MODEST_CLASS = 2 \n')
        outfile.write('AND s.RA > %.7f AND s.RA < %.7f \n' % (ra_min,ra_max))
        outfile.write('AND s.DEC > %.7f AND s.DEC < %.7f;' % (dec_min,dec_max))
        outfile.write('> %s \n' % db)
        outfile.close()

    ### def query(self,dbase,task,query):
    ###     import despydb.desdbi
    ###     logger.info("Running query...")
    ###  
    ###     desfile = os.path.join(os.getenv('HOME'),'.desservices.ini')
    ###     section = 'db-dessci'
    ###     dbi = despydb.desdbi.DesDbi(desfile,section)
    ###     cursor = dbi.cursor()
    ###     cursor.execute(open(query,'r').read())
    ###  
    ###     header = [d[0] for d in cursor.description]
    ###     data = cursor.fetchall()
    ###        
    ###     logger.info("Found %i rows."%len(data))
    ###     if not len(header) or not len(data): return None
    ###     array = np.rec.array(data,names=header)
    ###     return array

    def query(self,dbase,task,query):
        logger.info("Running query...")
        cmd = "easyaccess -l %s" % (query)
        logger.info(cmd)
        return subprocess.call(cmd,shell=True)
        #ret = subprocess.check_output(cmd,shell=True,stderr=subprocess.STDOUT) 
        #if 'ERROR:' in ret:
        #    raise subprocess.CalledProcessError(1,cmd,ret)
        #return ret

    def download(self, pixel, outdir=None, force=False):
        if outdir is None: outdir = './'
        else:              mkdir(outdir)
        sqldir = mkdir(os.path.join(outdir,'sql'))
        self._setup_desdbi()

        basename = self.basename + "_%04d"%pixel['name']
        sqlname = os.path.join(sqldir,basename+'.sql')
        taskname = basename
        outfile = os.path.join(outdir,basename+".fits")
        # ADW: There should be a 'force' option here
        if os.path.exists(outfile) and not force:
            logger.warning("Found %s; skipping..."%(outfile))
            return

        logger.info("\nDownloading pixel: %(name)i (ra=%(ra_min)g:%(ra_max)g,dec=%(dec_min)g:%(dec_max)g)"%(pixel))
        logger.info("Working on "+sqlname)
         
        self.generate_query(pixel['ra_min'],pixel['ra_max'],pixel['dec_min'],pixel['dec_max'],sqlname,outfile)
        ret = self.query(self.release,taskname,sqlname)
        if ret != 0:
            msg = "Download failed to complete."
            raise Exception(msg)
        return outfile

    def run(self,pixfile=None,outdir=None):
        self.load_pixels(pixfile)
        for pixel in self.pixels:
            self.download(pixel,outdir)

class Y1A1Database(DESDatabase):
    def __init__(self):
        release='Y1A1'
        super(Y1A1Database,self).__init__(release=release)
        self.release = release.lower()
        self.basename = "des_%s_photometry"%self.release
        print(self.basename)

    def generate_query(self, ra_min,ra_max,dec_min,dec_max,filename,db):
        # Preliminary and untested
        table = 'Y1A1_COADD_OBJECTS@DESSCI'

        select = 'ABS(s.WAVG_SPREAD_MODEL_I) < 0.004\n'
        select += 'AND s.FLAGS_G < 4 and s.FLAGS_R < 4 and s.FLAGS_I < 4\n'
        select += 'AND s.MAG_AUTO_G between 0 and 30 and s.MAG_AUTO_R between 0 and 30 and s.MAG_AUTO_I between 0 and 30\n'
        select += 'AND s.MAGERR_AUTO_G < 1 and s.MAGERR_AUTO_R < 1 and s.MAGERR_AUTO_I < 1'
        #select = '1 = 1'

        outfile = open(filename,"w")
        outfile.write('-- %s \n'%(self.__class__.__name__))
        outfile.write('SELECT s.COADD_OBJECTS_ID, s.RA, s.DEC, \n')
        outfile.write('s.MAG_PSF_G-XCORR_SFD98_G as MAGSFD_PSF_G, s.MAGERR_PSF_G, \n')
        outfile.write('s.MAG_PSF_R-XCORR_SFD98_R as MAGSFD_PSF_R, s.MAGERR_PSF_R, \n')
        outfile.write('s.MAG_PSF_I-XCORR_SFD98_I as MAGSFD_PSF_I, s.MAGERR_PSF_I, \n')
        outfile.write('s.MAG_AUTO_G-XCORR_SFD98_G as MAGSFD_AUTO_G, s.MAGERR_AUTO_G, \n')
        outfile.write('s.MAG_AUTO_R-XCORR_SFD98_R as MAGSFD_AUTO_R, s.MAGERR_AUTO_R, \n')
        outfile.write('s.MAG_AUTO_I-XCORR_SFD98_I as MAGSFD_AUTO_I, s.MAGERR_AUTO_I, \n')
        outfile.write('s.WAVG_SPREAD_MODEL_I, s.WAVG_SPREADERR_MODEL_I, \n')
        outfile.write('s.SPREAD_MODEL_I, s.SPREADERR_MODEL_I  \n')
        outfile.write('FROM %s s \n'%(table))
        outfile.write('WHERE %s \n'%(select))
        outfile.write('AND s.RA > %.7f AND s.RA < %.7f \n' % (ra_min,ra_max))
        outfile.write('AND s.DEC > %.7f AND s.DEC < %.7f; \n' % (dec_min,dec_max))
        outfile.write(' > %s \n' % (db))
        outfile.close()


class Y2NDatabase(DESDatabase):
    def __init__(self):
        release='Y2N'
        super(Y2NDatabase,self).__init__(release=release)

    def generate_query(self, ra_min,ra_max,dec_min,dec_max,filename,db):
        # Preliminary and untested
        table = 'kadrlica.Y2N_UNIQUE_OBJECTS@DESOPER'

        select = 'ABS(s.WAVG_SPREAD_MODEL_R) < 0.003\n'
        select += 'AND s.WAVG_MAG_PSF_G between 0 and 30 and s.WAVG_MAG_PSF_R between 0 and 30\n'
        select += 'AND s.MAGERR_AUTO_G < 1 and s.MAGERR_AUTO_R < 1'
        #select = '1 = 1'

        outfile = open(filename,"w")
        outfile.write('-- %s \n'%(self.__class__.__name__))
        outfile.write('SELECT s.CATALOG_ID, s.RA, s.DEC, \n')
        outfile.write('s.WAVG_MAG_PSF_G, s.MAGERR_PSF_G, \n')
        outfile.write('s.WAVG_MAG_PSF_R, s.MAGERR_PSF_R, \n')
        outfile.write('s.WAVG_MAG_AUTO_G, s.MAGERR_AUTO_G, \n')
        outfile.write('s.WAVG_MAG_AUTO_R, s.MAGERR_AUTO_R, \n')
        outfile.write('s.WAVG_SPREAD_MODEL_R, s.WAVG_SPREADERR_MODEL_R, \n')
        outfile.write('s.SPREAD_MODEL_R, s.SPREADERR_MODEL_R \n')
        outfile.write('FROM %s s \n'%(table))
        outfile.write('WHERE %s; \n'%(select))
        outfile.write(' > %s \n'%(db))
        outfile.close()

class Y2Q1Database(DESDatabase):
    def __init__(self):
        release='Y2Q1'
        super(Y2Q1Database,self).__init__(release=release)

    def generate_query(self, ra_min,ra_max,dec_min,dec_max,filename,db):
        # Preliminary and untested
        table = 'kadrlica.Y2Q1_OBJECTS_V1@DESOPER'

        select = 'ABS(s.WAVG_SPREAD_MODEL_R) < 0.003 + s.SPREADERR_MODEL_R \n'
        select += 'AND s.WAVG_MAG_PSF_G between 0 and 30 and s.WAVG_MAG_PSF_R between 0 and 30\n'
        select += 'AND s.WAVG_MAGERR_PSF_G < 1 and s.WAVG_MAGERR_PSF_R < 1'

        outfile = open(filename,"w")
        outfile.write('-- %s \n'%(self.__class__.__name__))
        outfile.write('SELECT s.CATALOG_ID, s.RA, s.DEC, \n')
        outfile.write('s.MAG_PSF_G, s.MAGERR_PSF_G, \n')
        outfile.write('s.WAVG_MAG_PSF_G, s.WAVG_MAGERR_PSF_G, \n')
        outfile.write('s.MAG_PSF_R, s.MAGERR_PSF_R, \n')
        outfile.write('s.WAVG_MAG_PSF_R, s.WAVG_MAGERR_PSF_R, \n')
        outfile.write('s.SPREAD_MODEL_R, s.SPREADERR_MODEL_R, \n')
        outfile.write('s.WAVG_SPREAD_MODEL_R, s.CLASS_STAR_R \n')
        outfile.write('FROM %s s \n'%(table))
        outfile.write('WHERE %s; \n'%(select))
        outfile.write(' > %s \n'%(db))
        outfile.close()

if __name__ == "__main__":
    import ugali.utils.parser
    description = "Download data set."
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_config()
    parser.add_debug()
    parser.add_verbose()
    parser.add_argument('pixfile',metavar='pixels.dat',default=None,
                        nargs='?',help='Input pixel file.')
    opts = parser.parse_args()

    from ugali.utils.config import Config
    config = Config(opts.config)

    survey = config['data']['survey'].lower()
    release = config['data']['release'].lower()
    outdir = config['data']['dirname']

    db = databaseFactory(survey,release)
    db.run(pixfile=opts.pixfile,outdir=outdir)
