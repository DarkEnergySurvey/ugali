import sys
import os
try: 
    from setuptools import setup
except ImportError: 
    from distutils.core import setup

here = os.path.abspath(os.path.dirname(__file__))

from ugali.version import get_git_version

def read(filename):
    return open(os.path.join(here,filename)).read()

setup(
    name='ugali',
    version=get_git_version(),
    url='https://bitbucket.org/bechtol/ugali/src',
    author='Keith Bechtol & Alex Drlica-Wagner',
    author_email='kadrlica@fnal.gov',
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
    packages=['ugali'],
    description="Ultra-faint galaxy likelihood fitting code.",
    long_description=read('README.md'),
    platforms='any',
    keywords=[''],
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
    ]
)
