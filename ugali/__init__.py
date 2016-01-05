"""
This is the Ultra-faint Galaxy Likelihood (UGaLi) software package.

"""
from .version import version
from analysis.isochrone import isochroneFactory
from analysis.kernel import kernelFactory

__author__ = "Keith Bechtol & Alex Drlica-Wagner"
__email__ = "bechtol@kicp.uchicago.edu, kadrlica@fnal.gov"
__version__ = version
__revision__ = ""
