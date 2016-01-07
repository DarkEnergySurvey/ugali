"""
This is the Ultra-faint Galaxy Likelihood (UGaLi) software package.

"""
__author__ = "Keith Bechtol & Alex Drlica-Wagner"
__email__ = "bechtol@kicp.uchicago.edu, kadrlica@fnal.gov"

try:
    from .version import __version__
except ImportError:
    from .versioning import get_version
    __version__ = get_version()

from analysis.isochrone import isochroneFactory
from analysis.kernel import kernelFactory

