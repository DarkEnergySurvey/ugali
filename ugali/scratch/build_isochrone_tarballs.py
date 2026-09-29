#!/usr/bin/env python
"""
Build the isochrone tarballs that get attached to a ugali release.

The isochrone libraries are too large to distribute with the source, so they
are attached to a GitHub release as one tarball per survey and model, plus a
small 'tiny' bundle that is installed by default. This script packs those
tarballs from a local isochrone directory (i.e. one that was filled in with
`download_isochrones.py`).

The layout inside each tarball must match what `setup.py` expects to extract
into `$UGALIDIR`:

    ugali-<survey>-<model>.tar.gz   ->  isochrones/<survey>/<model>/*.dat
    ugali-isochrones-tiny.tar.gz    ->  isochrones/<survey>/<model>/*.dat

Examples
--------
# Pack the libraries that ship for the first time in v1.9.0
%(prog)s --outdir release --survey lsst --survey roman --survey euclid

# Pack the default (tiny) bundle
%(prog)s --outdir release --tiny
"""
__author__ = "Peter Ferguson"

import os
import glob
import subprocess
import hashlib
from collections import OrderedDict as odict

from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

# Ages (Gyr) and metallicities (Z) that make up the tiny bundle. These are
# chosen to match the grid points that the unit tests use.
TINY_AGES = ['10.0', '12.0']
TINY_ZS = ['0.00010', '0.00020']

# Libraries that exist for each survey. DES, Pan-STARRS and SDSS carry all
# four models; Dartmouth (dotter2008) has no LSST, Roman or Euclid filter set
# and MIST (dotter2016) has no Euclid, so those combinations are absent.
# This doubles as the contents of the tiny bundle, so that a default install
# is usable by downstream packages that work in those filter systems (e.g.
# LSSTDESC/streamobs). Keep in sync with SURVEY_MODELS in setup.py.
LIBRARIES = odict([
    ('des'   , ['bressan2012', 'marigo2017', 'dotter2008', 'dotter2016']),
    ('ps1'   , ['bressan2012', 'marigo2017', 'dotter2008', 'dotter2016']),
    ('sdss'  , ['bressan2012', 'marigo2017', 'dotter2008', 'dotter2016']),
    ('lsst'  , ['bressan2012', 'marigo2017', 'dotter2016']),
    ('roman' , ['bressan2012', 'marigo2017', 'dotter2016']),
    ('euclid', ['bressan2012', 'marigo2017']),
])


def get_iso_dir():
    """ Root of the local isochrone libraries. """
    ugalidir = os.path.expandvars(os.getenv('UGALIDIR', '$HOME/.ugali'))
    return os.path.join(ugalidir, 'isochrones')


def compressor(njobs=1):
    """ Use pigz when it is available (and more than one job is requested). """
    if njobs > 1:
        try:
            subprocess.check_output(['pigz', '--version'],
                                    stderr=subprocess.STDOUT)
            # -n: do not store the filename/timestamp, so that repacking the
            # same files produces the same tarball
            return 'pigz -n -p %i' % njobs
        except (OSError, subprocess.CalledProcessError):
            logger.warning("pigz not found; falling back to gzip")
    return 'gzip -n'


def checksum(filename, blocksize=2 ** 20):
    md = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(blocksize), b''):
            md.update(block)
    return md.hexdigest()


def create_tarball(tarball, members, indir, njobs=1, force=False):
    """Create a tarball of isochrone files.

    Parameters
    ----------
    tarball : output tarball
    members : paths to include, relative to the parent of `indir`
    indir   : the local isochrone directory (the parent of `members`)
    njobs   : number of compression threads
    force   : overwrite an existing tarball

    Returns
    -------
    tarball : the output tarball
    """
    if os.path.exists(tarball) and not force:
        logger.info("Found %s; skipping..." % tarball)
        return tarball

    mkdir(os.path.dirname(tarball) or '.')

    # Members are given relative to the parent of the isochrone directory so
    # that they unpack as 'isochrones/<survey>/<model>'
    root = os.path.dirname(os.path.normpath(indir))
    # --sort and --mtime keep the tarball reproducible; -h dereferences the
    # symlinks that are sometimes used to alias one filter set to another
    cmd = ("tar -ch --sort=name --mtime=@0 --owner=0 --group=0 "
           "--numeric-owner -C %s %s | %s > %s") % (
               root, ' '.join(members), compressor(njobs), tarball)
    logger.debug(cmd)
    subprocess.check_call(cmd, shell=True)
    return tarball


def build_library(survey, model, indir, outdir, njobs=1, force=False):
    """ Pack the full library for one survey and model. """
    relpath = os.path.join(os.path.basename(os.path.normpath(indir)),
                           survey, model)
    path = os.path.join(os.path.dirname(os.path.normpath(indir)), relpath)
    files = glob.glob(os.path.join(path, 'iso_a*.dat'))
    if not files:
        logger.warning("No isochrones found: %s" % path)
        return None

    tarball = os.path.join(outdir, 'ugali-%s-%s.tar.gz' % (survey, model))
    logger.info("Packing %s (%i files)..." % (tarball, len(files)))
    return create_tarball(tarball, [relpath], indir, njobs, force)


def build_tiny(indir, outdir, njobs=1, force=False):
    """ Pack the small bundle that is installed by default. """
    isodir = os.path.basename(os.path.normpath(indir))
    members, missing = [], []
    for survey, models in LIBRARIES.items():
        for model in models:
            for age in TINY_AGES:
                for z in TINY_ZS:
                    basename = 'iso_a%s_z%s.dat' % (age, z)
                    relpath = os.path.join(isodir, survey, model, basename)
                    path = os.path.join(
                        os.path.dirname(os.path.normpath(indir)), relpath)
                    if not os.path.exists(path):
                        missing.append(relpath)
                    else:
                        members.append(relpath)

    for relpath in missing:
        logger.warning("Missing from tiny bundle: %s" % relpath)
    if missing:
        raise IOError("%i files missing from the tiny bundle" % len(missing))

    tarball = os.path.join(outdir, 'ugali-isochrones-tiny.tar.gz')
    logger.info("Packing %s (%i files)..." % (tarball, len(members)))
    return create_tarball(tarball, members, indir, njobs, force)


if __name__ == "__main__":
    import ugali.utils.parser
    description = "Build the isochrone tarballs for a ugali release"
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_verbose()
    parser.add_force()
    parser.add_argument('-i', '--indir', default=None,
                        help='local isochrone directory [default: $UGALIDIR/isochrones]')
    parser.add_argument('-o', '--outdir', default='release',
                        help='output directory for the tarballs')
    parser.add_argument('-s', '--survey', default=None, action='append',
                        help='survey to pack (may be repeated)')
    parser.add_argument('-k', '--model', default=None, action='append',
                        help='isochrone model to pack (may be repeated)')
    parser.add_argument('-t', '--tiny', action='store_true',
                        help='pack the tiny (default install) bundle')
    parser.add_argument('-n', '--njobs', default=1, type=int,
                        help='number of compression threads (uses pigz)')
    args = parser.parse_args()

    indir = args.indir if args.indir else get_iso_dir()
    if not os.path.isdir(indir):
        raise IOError("Isochrone directory not found: %s" % indir)
    logger.info("Reading isochrones from: %s" % indir)

    tarballs = []
    if args.tiny:
        tarballs.append(build_tiny(indir, args.outdir, args.njobs, args.force))

    # '--tiny' on its own only builds the tiny bundle; the full libraries are
    # built when they are requested, or when nothing in particular is.
    surveys = args.survey
    if surveys is None:
        surveys = [] if args.tiny else list(LIBRARIES.keys())

    for survey in surveys:
        models = args.model if args.model else LIBRARIES.get(survey,['marigo2017'])
        for model in models:
            tarball = build_library(survey, model, indir, args.outdir,
                                    args.njobs, args.force)
            if tarball: tarballs.append(tarball)

    print("\n%-40s %10s  %s" % ('tarball', 'size (MB)', 'sha256'))
    for tarball in tarballs:
        print("%-40s %10.1f  %s" % (os.path.basename(tarball),
                                    os.path.getsize(tarball) / 1024.**2,
                                    checksum(tarball)))
