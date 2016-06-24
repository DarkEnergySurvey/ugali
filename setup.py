import sys
import os
import urllib
try: 
    from setuptools import setup, find_packages
    from setuptools.command.build_ext import build_ext as _build_ext
except ImportError: 
    from distutils.core import setup
    from distutils.command.build_ext import build_ext as _build_ext
    def find_packages():
        return ['ugali','ugali.analysis','ugali.config','ugali.observation',
                'ugali.preprocess','ugali.simulation','ugali.candidate',
                'ugali.utils']

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

def read(filename):
    return open(os.path.join(HERE,filename)).read()

CMDCLASS = versioneer.get_cmdclass()
        
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
