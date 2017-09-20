from __future__ import print_function

import sys
import os
import io

try: 
    from setuptools import setup, find_packages
    from setuptools.command.install import install as _install
except ImportError: 
    from distutils.core import setup
    from distutils.command.install import install as _install
    def find_packages():
        return ['ugali','ugali.analysis','ugali.config','ugali.observation',
                'ugali.preprocess','ugali.simulation','ugali.candidate',
                'ugali.utils']

import distutils.cmd

import versioneer
VERSION = versioneer.get_version()

NAME = 'ugali'
HERE = os.path.abspath(os.path.dirname(__file__))
URL = 'https://github.com/DarkEnergySurvey/ugali'
DESC = "Ultra-faint galaxy likelihood toolkit."
LONG_DESC = "See %s"%URL
CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Natural Language :: English
Operating System :: MacOS :: MacOS X
Operating System :: POSIX :: Linux
Programming Language :: Python :: 2.7
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Astronomy
Topic :: Scientific/Engineering :: Physics
"""

RELEASE = URL+'/releases/download/v1.7.0'
UGALIDIR = os.getenv("UGALIDIR","$HOME/.ugali")
ISOSIZE = "~1MB" 
CATSIZE = "~20MB"
# Could find file size dynamically, but it's a bit slow...
# int(urllib.urlopen(ISOCHRONES).info().getheaders("Content-Length")[0])/1024**2
SURVEYS = ['des','ps1','sdss']
MODELS = ['bressan2012','marigo2017','dotter2008','dotter2016']

class ProgressFileIO(io.FileIO):
    def __init__(self, path, *args, **kwargs):
        self._total_size = os.path.getsize(path)
        io.FileIO.__init__(self, path, *args, **kwargs)

    def read(self, size):
        count = self.tell()/size
        self.progress_bar(count,size,self._total_size)
        return io.FileIO.read(self, size)

    @staticmethod
    def progress_bar(count, block_size, total_size):
        block = 100*block_size/float(total_size)
        progress = count*block
        if progress % 1 < 1.01*block:
            msg = '\r[{:51}] ({:d}%)'.format(int(progress//2)*'='+'>',int(progress))
            sys.stdout.write(msg)
            sys.stdout.flush()

class TarballCommand(distutils.cmd.Command,object):
    """ Command for downloading data files """
    description = "install data files"
    user_options = [
        ('ugali-dir=',None,
         'path to install data files [default: %s]'%UGALIDIR),
        ('force','f',
         'force installation (overwrite any existing files)')
        ]
    boolean_options = ['force']
    release = RELEASE
    _tarball = None
    _dirname = None

    def initialize_options(self):
        self.ugali_dir = os.path.expandvars(UGALIDIR)
        self.force = False
        # Not really the best way, but ok...
        self.tarball = self._tarball
        self.dirname = self._dirname
        
    def finalize_options(self):
        # Required by abstract base class
        pass

    @property
    def path(self):
        return os.path.join(self.ugali_dir,self.dirname)

    def check_exists(self):
        return os.path.exists(self.path)
        
    def install_tarball(self, tarball):
        import urllib
        import tarfile

        if not os.path.exists(self.ugali_dir):
            print("creating %s"%self.ugali_dir)
            os.makedirs(self.ugali_dir)
        os.chdir(self.ugali_dir)

        url = os.path.join(self.release,tarball)

        print("downloading %s..."%url)
        if urllib.urlopen(url).getcode() >= 400:
            raise Exception('url does not exist')

        urllib.urlretrieve(url,tarball,reporthook=ProgressFileIO.progress_bar)
        print('')
        if not os.path.exists(tarball):
            raise urllib.error.HTTPError()
            
        print("extracting %s..."%tarball)
        with tarfile.open(fileobj=ProgressFileIO(tarball),mode='r:gz') as tar:
            ## Check if the directory exists?
            #if os.path.exists(tar.next().name) and not self.force:
            #    print("directory found; skipping installation")
            tar.extractall()
            tar.close()
            print('')

        print("removing %s"%tarball)
        os.remove(tarball)

    def run(self):
        if self.dry_run:
            print("skipping data install")
            return
        
        if self.check_exists():
            print("found %s"%self.path)
            if self.force:
                print("overwriting directory")
            else:
                print("use '--force' to overwrite")
                return
       
        self.install_tarball(self.tarball)

class CatalogCommand(TarballCommand):
    """ Command for downloading catalog files """
    description = "install catalog files"
    _tarball = 'ugali-catalogs.tar.gz'
    _dirname = 'catalogs'

class IsochroneCommand(TarballCommand):
    """ Command for downloading isochrone files """
    description = "install isochrone files"
    user_options = TarballCommand.user_options + [
        ('survey=',None,
         'survey set [default: None]'),
        ('model=',None,
         'isochrone model [default: None]')
        ]
    _tarball = 'ugali-isochrones-tiny.tar.gz'
    _dirname = 'isochrones'

    def initialize_options(self):
        super(IsochroneCommand,self).initialize_options()
        self.survey = None
        self.model = None

    def finalize_options(self):
        super(IsochroneCommand,self).finalize_options()
        self._build_surveys()
        self._build_models()

    def _build_surveys(self):
        if self.survey is None:
            self.surveys = SURVEYS
        else:
            self.survey = self.survey.lower()
            if self.survey not in SURVEYS:
                raise Exception("unrecognized survey: '%s'"%self.survey)
            self.surveys = [self.survey]

    def _build_models(self):
        if self.model is None: 
            self.models = MODELS
        else:
            self.model = self.model.lower()
            if self.model not in MODELS:
                raise Exception("unrecognized model: '%s'"%self.model)
            self.models = [self.model]

    def run(self):
        if self.dry_run:
            print("skipping data install")
            return

        if (self.survey is None) and (self.model is None):
            self.tarball = self._tarball
            self.dirname = self._dirname
            super(IsochroneCommand,self).run()
            return
        
        for survey in self.surveys:
            for model in self.models:
                self.tarball = "ugali-%s-%s.tar.gz"%(survey,model)
                self.dirname = "isochrones/%s/%s"%(survey,model)
                super(IsochroneCommand,self).run()


class install(_install):
    """ 
    Subclass the setuptools 'install' class.
    """
    user_options = _install.user_options + [
        ('isochrones',None,"install isochrone files (%s)"%ISOSIZE),
        ('catalogs',None,"install catalog files (%s)"%CATSIZE),
        ('ugali-dir=',None,"install file directory [default: %s]"%UGALIDIR),
    ]
    boolean_options = _install.boolean_options + ['isochrones','catalogs']

    def initialize_options(self):
        _install.initialize_options(self)
        self.ugali_dir = os.path.expandvars(UGALIDIR)
        self.isochrones = False
        self.catalogs = False

    def run(self):
        # run superclass install
        _install.run(self)

        # Could ask user whether they want to install isochrones, but 
        # pip filters sys.stdout, so the prompt never gets sent:
        # https://github.com/pypa/pip/issues/2732#issuecomment-97119093

        if self.isochrones: 
            self.install_isochrones()

        if self.catalogs: 
            self.install_catalogs()
            
    def install_isochrones(self):
        """
        Call to isochrone install command:
        http://stackoverflow.com/a/24353921/4075339
        """
        cmd_obj = self.distribution.get_command_obj('isochrones')
        cmd_obj.force = self.force
        if self.ugali_dir: cmd_obj.ugali_dir = self.ugali_dir
        self.run_command('isochrones')

    def install_catalogs(self):
        """
        Call to catalog install command:
        http://stackoverflow.com/a/24353921/4075339
        """
        cmd_obj = self.distribution.get_command_obj('catalogs')
        cmd_obj.force = self.force
        if self.ugali_dir: cmd_obj.ugali_dir = self.ugali_dir
        self.run_command('catalogs')
            
CMDCLASS = versioneer.get_cmdclass()
CMDCLASS['isochrones'] = IsochroneCommand
CMDCLASS['catalogs'] = CatalogCommand
CMDCLASS['install'] = install

setup(
    name=NAME,
    version=VERSION,
    cmdclass=CMDCLASS,
    url=URL,
    author='Keith Bechtol & Alex Drlica-Wagner',
    author_email='bechtol@kicp.uchicago.edu, kadrlica@fnal.gov',
    scripts = [],
    install_requires=[
        'numpy >= 1.9.0',
        'scipy >= 0.14.0',
        'healpy >= 1.6.0',
        'pyfits >= 3.1',
        'emcee >= 2.1.0',
        'corner >= 1.0.0',
        'pyyaml >= 3.10',
        # Add astropy, fitsio, matplotlib, ...
    ],
    packages=find_packages(),
    description=DESC,
    long_description=LONG_DESC,
    platforms='any',
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f]
)
