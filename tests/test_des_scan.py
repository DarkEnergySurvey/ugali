#!/usr/bin/env python
"""
Regression test: regenerate a DES Y6 likelihood scan pixel and compare
bit-for-bit against a reference output.

This requires the DES Y6 reference run (catalog/mask healpix files,
config, and reference scan files) and the matching isochrones, so it
is skipped unless the reference directory is available. Point to it
with the UGALI_DES_REFDIR environment variable (default: Chin-Yi's
run on the UW astro machines).

    UGALI_DES_REFDIR=/astro/users/chinyi/des_ugali pytest tests/test_des_scan.py -s

To test additional pixels, set UGALI_DES_PIXELS to a comma-separated
list of nside=256 pixel indices that exist in <refdir>/scan-v0.
"""
import os
import glob
import shutil
import subprocess

import numpy as np
import pytest

import ugali.analysis.scan

REFDIR = os.environ.get('UGALI_DES_REFDIR', '/astro/users/chinyi/des_ugali')
SCANDIR = os.path.join(REFDIR, 'scan-v0')
CONFIG = os.path.join(SCANDIR, 'config.yaml')
PIXELS = [int(p) for p in os.environ.get('UGALI_DES_PIXELS', '639648').split(',')]
NSIDE = 256

# Header keys that may legitimately differ between runs
IGNORE_KEYS = {'DATE', 'CHECKSUM', 'DATASUM', 'COMMENT', 'HISTORY'}

pytestmark = pytest.mark.skipif(not os.path.exists(CONFIG),
                                reason="DES Y6 reference run not found: %s" % CONFIG)


def compare_fits(ref, new):
    """Return a list of human-readable differences (empty if identical)."""
    import fitsio
    diffs = []
    fr, fn = fitsio.FITS(ref), fitsio.FITS(new)
    if len(fr) != len(fn):
        return ["HDU count %i != %i" % (len(fr), len(fn))]
    for i in range(len(fr)):
        hr, hn = fr[i].read_header(), fn[i].read_header()
        kr = {k: hr[k] for k in hr.keys() if k not in IGNORE_KEYS}
        kn = {k: hn[k] for k in hn.keys() if k not in IGNORE_KEYS}
        for k in sorted(set(kr) | set(kn)):
            if kr.get(k) != kn.get(k):
                diffs.append("HDU%i header %s: %r != %r" % (i, k, kr.get(k), kn.get(k)))
        if fr[i].get_exttype() == 'IMAGE_HDU':
            continue
        dr, dn = fr[i].read(), fn[i].read()
        if dr.dtype != dn.dtype or dr.shape != dn.shape:
            diffs.append("HDU%i dtype/shape %s%s != %s%s" % (i, dr.dtype, dr.shape, dn.dtype, dn.shape))
            continue
        for name in dr.dtype.names:
            a, b = dr[name], dn[name]
            same = np.array_equal(a, b, equal_nan=True) if a.dtype.kind == 'f' else np.array_equal(a, b)
            if not same:
                msg = "HDU%i column %s differs" % (i, name)
                if a.dtype.kind == 'f':
                    d = np.abs(a.astype(float) - b.astype(float))
                    msg += " (n=%i/%i, max|diff|=%.3g)" % (np.sum(d > 0), d.size, np.nanmax(d))
                diffs.append(msg)
    return diffs


@pytest.mark.parametrize('pix', PIXELS)
def test_scan_matches_reference(pix, tmp_path):
    """Regenerate one likelihood-scan pixel and compare with the reference."""
    reffile = os.path.join(SCANDIR, 'scan_%08i_cel.fits' % pix)
    assert os.path.exists(reffile), "Missing reference file: %s" % reffile

    # Run from a scratch directory with the reference config (output
    # paths in the config are relative).
    workdir = tmp_path / 'des_scan'
    workdir.mkdir()
    shutil.copy(CONFIG, workdir / 'config.yaml')
    outfile = workdir / ('scan_%08i_cel.fits' % pix)

    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    cmd = ['python', ugali.analysis.scan.__file__, 'config.yaml', str(outfile),
           '--hpx', str(NSIDE), str(pix)]
    subprocess.check_call(cmd, cwd=str(workdir), env=env)
    assert outfile.exists()

    diffs = compare_fits(reffile, str(outfile))
    assert not diffs, "scan_%08i_cel.fits differs from reference:\n  " % pix + "\n  ".join(diffs)
