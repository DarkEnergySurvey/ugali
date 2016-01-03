#!/usr/bin/env python

version_tag = (1,5,0)
version = '.'.join(map(str, version_tag[:3]))

if len(version_tag) > 3:
    version += version_tag[3]

def get_git_version():
    """
    Eventually we may want to do something like this.
    """
    from subprocess import check_output

    PREFIX = 'v'
    cmd = 'git describe --tags --match %s[0-9]*' % PREFIX
    try:
        version = check_output(cmd.split()).decode().strip()[len(PREFIX):]
    except CalledProcessError:
        raise RuntimeError('Unable to get version number from git tags')

    if '-' in version:
        version = version.split('-')[0] + '.dev'

    return version


if __name__ == "__main__":
    print version
