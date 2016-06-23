"""
This is the Ultra-faint Galaxy Likelihood (UGaLi) software package.

"""
__author__ = "Keith Bechtol & Alex Drlica-Wagner"
__email__ = "bechtol@kicp.uchicago.edu, kadrlica@fnal.gov"


# Automatically grab the version from the git tag to ensure that the
# tag and the code agree. For more details, see `get_version.py`
#try:
#    from .version import __version__
#except ImportError:
#    from .get_version import get_version
#    __version__ = get_version()

from analysis.isochrone import isochroneFactory
from analysis.kernel import kernelFactory

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
