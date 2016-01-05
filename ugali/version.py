# Modified from https://github.com/Changaco/version.py

from os.path import abspath, dirname, isdir, join
import re
from subprocess import CalledProcessError, check_output

PREFIX = 'v'
tag_re = re.compile(r'\btag: %s([0-9][^,]*)\b' % PREFIX)
version_re = re.compile('^Version: (.+)$', re.M)

def get_version():
    # Default value
    version = 'dev'

    # Return the version if it has been injected into the file by git-archive
    version = tag_re.search('$Format:%D$')
    if version:
        return version.group(1)

    try: 
        out = check_output('git rev-parse'.split())
        isgit = True
    except CalledProcessError:
        isgit = False

    if isgit:
        # Get the version using "git describe".
        cmd = 'git describe --tags --match %s[0-9]*' % PREFIX
        try:
            version = check_output(cmd.split()).decode().strip()[len(PREFIX):]
        except CalledProcessError:
            raise RuntimeError('Unable to get version from git tag')

    # PEP 440 compatibility
    if '-' in version:
        version = '.dev'.join(version.split('-')[:2])

    return version

if __name__ == '__main__':
    print get_version()

### #!/usr/bin/env python
### import os, sys
###  
### # This is a static release and shouldn't be trusted
### RELEASE = '1.5.1'
###  
### if .os
###  
### def get_version():
###     if os
###  
### def get_git_version():
###     """
###     Eventually we may want to do something like this.
###     """
###     from subprocess import check_output
###  
###     PREFIX = 'v'
###     cmd = 'git describe --tags --match %s[0-9]*' % PREFIX
###     try:
###         version = check_output(cmd.split()).decode().strip()[len(PREFIX):]
###     except CalledProcessError:
###         raise RuntimeError('Unable to get version number from git tags')
###  
###     if '-' in version:
###         version = version.split('-')[0] + '.dev'
###  
###     return version
###  
### if __name__ == "__main__":
###     print version
