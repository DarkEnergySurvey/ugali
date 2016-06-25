import sys
import os
import io

try: 
    from setuptools import setup, find_packages
except ImportError: 
    from distutils.core import setup
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
CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
Programming Language :: Python
Natural Language :: English
Topic :: Scientific/Engineering
"""
ISOCHRONES = URL+'/releases/download/v1.5.2/ugali-isochrones-v1.5.2.tar.gz'

def read(filename):
    return open(os.path.join(HERE,filename)).read()

def progress_bar(count, block_size, total_size):
    block = 100*block_size/float(total_size)
    progress = count*block
    if progress % 1 < 1.01*block:
        msg = '[{:51}] ({:d}%)\r'.format(int(progress//2)*'='+'>',int(progress))
        #msg = '({:d}%)\r'.format(int(progress))
        sys.stdout.write(msg)
        sys.stdout.flush()
    #percent = int(100*count*block_size/total_size)
    

class ProgressFileObject(io.FileIO):
    def __init__(self, path, *args, **kwargs):
        self._total_size = os.path.getsize(path)
        io.FileIO.__init__(self, path, *args, **kwargs)

    def read(self, size):
        count = self.tell()/size
        progress_bar(count,size,self._total_size)
        return io.FileIO.read(self, size)

class IsochroneCommand(distutils.cmd.Command):
    """ Command for downloading isochrone files """
    description = "install isochrone files"
    user_options = [
        ('isochrone-path=',None,
         'path to install isochrones (default: $HOME/.ugali)'),
        ('no-isochrone',None,
         'do not download and install isochrones (default: False)')
        ]
    boolean_options = ['no-isochrone']

    def initialize_options(self):
        self.isochrone_path = os.path.expandvars('$HOME/.ugali')
        self.no_isochrone = False

    def finalize_options(self):
        pass

    def install_isochrones(self):
        import urllib
        import tarfile

        print("installing isochrones")
        print("creating %s"%self.isochrone_path)
        if not os.path.exists(self.isochrone_path):
            os.makedirs(self.isochrone_path)

        os.chdir(self.isochrone_path)

        url = ISOCHRONES
        tarball = os.path.basename(url)

        print("downloading %s"%url)
        urllib.urlretrieve(url, tarball, reporthook=progress_bar)
        print('')

        print("extracting %s"%tarball)
        with tarfile.open(fileobj=ProgressFileObject(tarball),mode='r:gz') as tar:
            tar.extractall()
            tar.close()
            print('')

        print("removing %s"%tarball)
        os.remove(tarball)

    def run(self):
        if self.no_isochrone:
            print("skipping isochrone download")
            return

        if os.path.exists(os.path.join(self.isochrone_path,'isochrones')):
            print("isochrone directory found; skipping")
            return

        self.install_isochrones()

CMDCLASS = versioneer.get_cmdclass()
CMDCLASS['isochrones'] = IsochroneCommand

setup(
    name=NAME,
    version=VERSION,
    cmdclass=CMDCLASS,
    url=URL,
    author='Keith Bechtol & Alex Drlica-Wagner',
    author_email='bechtol@kicp.uchicago.edu, kadrlica@fnal.gov',
    scripts = [],
    install_requires=[
        'python >= 2.7.0',
        'numpy >= 1.9.0',
        'scipy >= 0.14.0',
        'healpy >= 1.6.0',
        'pyfits >= 3.1',
        'emcee >= 2.1.0',
        'pyyaml >= 3.10',
    ],
    packages=find_packages(),
    package_data={'ugali': ['data/catalog.tgz']},
    description="Ultra-faint galaxy likelihood toolkit.",
    long_description=read('README.md'),
    platforms='any',
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f]
)
