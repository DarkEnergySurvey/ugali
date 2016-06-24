import sys
import os
import io
import urllib
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
    percent = int(100*count*block_size/total_size)
    msg = '[{:51}] ({:d}%)\r'.format(percent//2*'='+'>',percent)
    sys.stdout.write(msg)
    sys.stdout.flush()

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
        import urllib2
        import tarfile

        print("installing isochrones")
        print("creating %s"%self.isochrone_path)
        if not os.path.exists(self.isochrone_path):
            os.makedirs(self.isochrone_path)

        url = ISOCHRONES
        basename = os.path.basename(url)
        tarball = os.path.join(self.isochrone_path,basename)

        print("downloading %s"%url)
        urllib.urlretrieve(url, tarball, reporthook=progress_bar)
        print('')

        print("extracting %s"%basename)
        with tarfile.open(fileobj=ProgressFileObject(tarball),mode='r:gz') as tar:
            tar.extractall()
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


"""
import tarfile
import io
import os

def get_file_progress_file_object_class(on_progress):
    class FileProgressFileObject(tarfile.ExFileObject):
        def read(self, size, *args):
          on_progress(self.name, self.position, self.size)
          return tarfile.ExFileObject.read(self, size, *args)
    return FileProgressFileObject

class TestFileProgressFileObject(tarfile.ExFileObject):
    def read(self, size, *args):
      on_progress(self.name, self.position, self.size)
      return tarfile.ExFileObject.read(self, size, *args)

class ProgressFileObject(io.FileIO):
    def __init__(self, path, *args, **kwargs):
        self._total_size = os.path.getsize(path)
        io.FileIO.__init__(self, path, *args, **kwargs)

    def read(self, size):
        print("Overall process: %d of %d" %(self.tell(), self._total_size))
        return io.FileIO.read(self, size)

def on_progress(filename, position, total_size):
    print("%s: %d of %s" %(filename, position, total_size))

tarfile.TarFile.fileobject = get_file_progress_file_object_class(on_progress)
tar = tarfile.open(fileobj=ProgressFileObject("a.tgz"))
tar.extractall()
tar.close()
"""

"""
import urllib2, sys

def chunk_report(bytes_so_far, chunk_size, total_size):
   percent = float(bytes_so_far) / total_size
   percent = round(percent*100, 2)
   sys.stdout.write("Downloaded %d of %d bytes (%0.2f%%)\r" % 
       (bytes_so_far, total_size, percent))

   if bytes_so_far >= total_size:
      sys.stdout.write('\n')

def chunk_read(response, chunk_size=8192, report_hook=None):
   total_size = response.info().getheader('Content-Length').strip()
   total_size = int(total_size)
   bytes_so_far = 0

   while 1:
      chunk = response.read(chunk_size)
      bytes_so_far += len(chunk)

      if not chunk:
         break

      if report_hook:
         report_hook(bytes_so_far, chunk_size, total_size)

   return bytes_so_far

if __name__ == '__main__':
   response = urllib2.urlopen('http://www.ebay.com');
   chunk_read(response, report_hook=chunk_report)
"""
