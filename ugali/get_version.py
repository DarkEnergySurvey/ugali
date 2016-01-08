#!/usr/bin/env python
"""
Script for automatically generating version numbers with each commit.

References:
https://github.com/Changaco/version.py
http://www.mass-communicating.com/code/2013/11/08/python-versions.html
"""
import sys
import re
from subprocess import CalledProcessError, check_output

__all__ = ['get_version','write_version_py']

ARCHIVE = '$Format:%d$'
DEFAULT = 'none'
PREFIX  = 'v'
TEMPLATE = """\
# FILE GENERATED ON GIT COMMIT
__version__ = '%(version)s'
"""

def get_version():
    """
    Get the version string from git. First tries `export-subst` value
    from `git-archive` and then `git-describe` value.
    """
    # Default value
    version = DEFAULT

    # Return the version if it has been injected into the file by git-archive
    tag_re = re.compile(r'\btag: %s([0-9][^,]*)\b' % PREFIX)
    if tag_re.search(ARCHIVE):
        version = tag_re.search(ARCHIVE)
        return version.group(1)
    elif 'HEAD' in ARCHIVE:
        version = 'HEAD'
        return version

    # Return tag if directory is git controlled
    cmd = 'git describe --tags --match %s[0-9]*' % PREFIX
    try: 
        version = check_output(cmd.split()).decode().strip()[len(PREFIX):]
        # PEP 440 compatibility
        if '-' in version:
            version = '.dev'.join(version.split('-')[:2])
        return version
    except CalledProcessError:
        raise RuntimeError('Unable to get version from git tag')

    return version

def write_version_py(filename='ugali/version.py',**kwargs):
    out = open(filename,'w')
    out.write(TEMPLATE%kwargs)
    out.close()

if __name__ == "__main__":
    import argparse
    description = "Script for parsing git version"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--clean',action='store_true')
    parser.add_argument('-n','--dry-run',action='store_true')
    args = parser.parse_args()
    
    version = '' if args.clean else get_version()
    if args.dry_run: 
        print TEMPLATE%dict(version=version)
    else:
        write_version_py('ugali/version.py', version=version)
