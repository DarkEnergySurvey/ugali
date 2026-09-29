#!/usr/bin/env python
"""
Build a MIST (Dotter2016) isochrone library with one request per metallicity.

`download_isochrones.py` asks the server for one isochrone at a time, which
for a MIST library means 12600 requests (126 ages x 100 metallicities).
mist.science does not tolerate that: it degrades until it answers every query
with "failed in age interpolation!" and only recovers after being left alone
for the better part of an hour.

The MIST form can return a whole age sequence in a single request
(`age_type=range`), so a metallicity's 126 ages cost one query instead of 126
-- a 126x reduction, and the age interpolation is still done by MIST rather
than reimplemented here. The result is the same product: compared against
isochrones downloaded one at a time, the magnitudes are bit-identical and the
theory columns agree to ~1e-14 (the server rounds its age sequence slightly
differently).

Examples
--------
# One library
%(prog)s -s lsst -o $UGALIDIR/isochrones/lsst/dotter2016

# Both, with a gentler pace between requests
%(prog)s -s lsst -s roman --delay 20
"""
__author__ = "Peter Ferguson"

import os
import re
import sys
import time
import shutil
import zipfile
import tempfile
import contextlib
import datetime

try:
    from urllib.parse import urlencode
    from urllib.request import urlopen
except ImportError:
    from urllib import urlencode
    from urllib2 import urlopen

import numpy as np

from ugali.utils.logger import logger
from ugali.utils.shell import mkdir
from ugali.isochrone.mesa import Dotter2016, dict_output

SERVER = 'https://mist.science'


def log(msg):
    stamp = datetime.datetime.now().strftime('%H:%M:%S')
    print('%s  %s' % (stamp, msg))
    sys.stdout.flush()


def request_range(survey, feh, ages):
    """ Ask for every age at one metallicity; return the photometry file text.

    Parameters
    ----------
    survey : ugali survey name
    feh    : [Fe/H] of the isochrones
    ages   : ages (Gyr) wanted; only the endpoints and spacing are sent

    Returns
    -------
    text : contents of the '.iso.<system>' file in the returned archive
    """
    output = dict_output[survey]
    params = dict(version='MIST1', v_div_vcrit='vvcrit0.4',
                  age_scale='linear', age_type='range',
                  age_range_low=ages.min() * 1e9,
                  age_range_high=ages.max() * 1e9,
                  age_range_delta=(ages[1] - ages[0]) * 1e9,
                  FeH_value=feh, alpha_value='p0',
                  output_option='photometry', output=output, Av_value=0)

    query = urlencode(params).encode('utf-8')
    with contextlib.closing(urlopen(SERVER + '/iso_form.php', query)) as r:
        response = r.read().decode('utf-8', errors='replace')

    match = re.search(r'href="([^"]+\.zip)"', response)
    if match is None:
        raise RuntimeError('Output filename not found: %s'
                           % response.strip()[:120])
    href = match.group(1)

    tmpdir = tempfile.mkdtemp()
    try:
        archive = os.path.join(tmpdir, os.path.basename(href))
        url = '%s/%s' % (SERVER, href.lstrip('/'))
        with contextlib.closing(urlopen(url)) as r:
            with open(archive, 'wb') as tmp:
                shutil.copyfileobj(r, tmp)
        with zipfile.ZipFile(archive) as zf:
            suffix = '.iso.%s' % output
            members = [n for n in zf.namelist() if n.endswith(suffix)]
            if not members:
                raise RuntimeError("No '%s' in %s" % (suffix, href))
            with zf.open(members[0]) as fh:
                return fh.read().decode('utf-8', errors='replace')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def split_isochrones(text, ages):
    """ Split a multi-isochrone file into (age, lines) for each isochrone.

    Each isochrone is written back with the file's own global header, so the
    result is byte-for-byte what the server returns when asked for that
    single age.
    """
    lines = text.splitlines(True)
    first = next(i for i, l in enumerate(lines)
                 if l.startswith('# number of EEPs'))
    # The header advertises how many isochrones follow; each split file has 1.
    # The count is right-aligned in a fixed six-character field, so it has to
    # be rewritten at that width for the result to match the file the server
    # returns when asked for a single age.
    header = [re.sub(r'(number of isochrones\s*=)\s*\d+',
                     lambda m: '%s%6d' % (m.group(1), 1), l)
              for l in lines[:first]]

    out, i = [], first
    while i < len(lines):
        if not lines[i].startswith('# number of EEPs'):
            i += 1
            continue
        nrow = int(lines[i].split('=')[1].split()[0])
        block = lines[i:i + 3 + nrow]
        # The age column is log10(yr) in this format despite its name
        value = float(block[3].split()[1])
        age = 10 ** value / 1e9 if value < 100 else value / 1e9
        # Snap to the requested grid so the filename is exact
        idx = np.argmin(np.abs(ages - age))
        if abs(ages[idx] - age) > 0.01:
            raise RuntimeError('Unexpected age %g Gyr in response' % age)
        out.append((ages[idx], header + block))
        i += 3 + nrow
    return out


def build(survey, outdir, delay=5.0, force=False,
          backoff=900.0, max_backoff=3600.0, tries=6):
    """ Download and unpack a whole Dotter2016 library.

    mist.science grants a burst of a few heavy requests and then refuses
    everything for the best part of an hour before granting another. Marching
    on through the remaining metallicities during a lockout just wastes
    requests -- it produced ~95 failures for every 5 successes -- so on a
    failure this waits on the *same* metallicity, backing off, until the
    server lets it through again.

    Parameters
    ----------
    survey      : ugali survey name
    outdir      : where to write the library
    delay       : seconds between successful requests
    backoff     : first wait after a refusal, doubling up to max_backoff
    tries       : attempts per metallicity before giving up on it this pass
    """
    bands = Dotter2016.band_names[survey]
    iso = Dotter2016(survey=survey, band_1=bands[0], band_2=bands[1],
                     age=12.0, metallicity=2e-4, distance_modulus=18)
    ages, zs = iso.abins, iso.zbins
    mkdir(outdir)

    wrote = failed = 0
    for n, z in enumerate(zs, 1):
        names = [iso.params2filename(a, z) for a in ages]
        if not force and all(os.path.exists(os.path.join(outdir, f))
                             for f in names):
            continue

        feh = Dotter2016.z2feh(z)
        wait = backoff
        for attempt in range(1, tries + 1):
            try:
                text = request_range(survey, feh, ages)
                blocks = split_isochrones(text, ages)
                if len(blocks) != len(ages):
                    raise RuntimeError('got %d isochrones, expected %d'
                                       % (len(blocks), len(ages)))
            except Exception as e:
                if attempt == tries:
                    logger.warning('Z=%.5f ([Fe/H]=%+.3f): giving up -- %s'
                                   % (z, feh, e))
                    failed += 1
                    break
                log('%s Z=%.5f refused (%s); waiting %.0fs [try %d/%d]'
                    % (survey, z, str(e).split(':')[0], wait, attempt, tries))
                time.sleep(wait)
                wait = min(wait * 2, max_backoff)
                continue

            for age, content in blocks:
                path = os.path.join(outdir, iso.params2filename(age, z))
                with open(path, 'w') as f:
                    f.writelines(content)
                wrote += 1
            log('%s Z=%.5f ([Fe/H]=%+.3f) %3d/%d -- %d files (%d written, %d failed)'
                % (survey, z, feh, n, len(zs), len(blocks), wrote, failed))
            time.sleep(delay)
            break

    have = len(os.listdir(outdir))
    log('%s: %d/%d files on disk (%d metallicities failed)'
        % (survey, have, len(ages) * len(zs), failed))
    return have, len(ages) * len(zs), failed


if __name__ == "__main__":
    import ugali.utils.parser
    parser = ugali.utils.parser.Parser(description=__doc__)
    parser.add_verbose()
    parser.add_force()
    parser.add_argument('-s', '--survey', action='append', default=None,
                        help='survey to build (repeatable)')
    parser.add_argument('-o', '--outdir', default=None,
                        help='output directory (default: $UGALIDIR tree)')
    parser.add_argument('--delay', default=5.0, type=float,
                        help='seconds between successful requests')
    parser.add_argument('--backoff', default=900.0, type=float,
                        help='first wait after the server refuses, doubling')
    parser.add_argument('--passes', default=5, type=int,
                        help='sweeps to make, to pick up transient failures')
    args = parser.parse_args()

    surveys = args.survey if args.survey else ['lsst', 'roman']
    from ugali.isochrone.model import get_iso_dir

    for survey in surveys:
        outdir = args.outdir or os.path.join(get_iso_dir(), survey, 'dotter2016')
        for i in range(1, args.passes + 1):
            have, want, failed = build(survey, outdir, args.delay,
                                       args.force, args.backoff)
            if have >= want:
                break
            log('%s: pass %d left %d missing; retrying' % (survey, i, want - have))
            time.sleep(60)
