#!/usr/bin/env python
from subprocess import check_output

__all__ = ("get_git_version")

def get_git_version():
    PREFIX = 'v'
    cmd = 'git describe --tags --match %s[0-9]*' % PREFIX
    try:
        version = check_output(cmd.split()).decode().strip()[len(PREFIX):]
    except CalledProcessError:
        raise RuntimeError('Unable to get version number from git tags')

    if '-' in version:
        version = version.split('-')[0] + '-dev'

    return version

if __name__ == "__main__":
    print get_git_version()
