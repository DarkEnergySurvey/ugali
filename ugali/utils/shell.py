#!/usr/bin/env python
import shutil
import os
import errno    

# Tools for working with the shell

def pwd():
    # Careful, won't work after a call to os.chdir...
    return os.environ['PWD']

def mkdir(path):
    # https://stackoverflow.com/a/600612/4075339
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    return path

def mkscratch():
    if os.path.exists('/scratch/'):    
        return(mkdir('/scratch/%s/'%os.environ['USER']))
    elif os.path.exists('/tmp/'):
        return(mkdir('/tmp/%s/'%os.environ['USER']))
    else:
        raise Exception('...')

def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def get_ugali_dir():
    """Get the path to the ugali data directory from the environment"""

    ugalidir = os.getenv('UGALIDIR')

    # Get the HOME directory
    if not ugalidir:
        ugalidir=os.path.join(os.getenv('HOME'),'.ugali')

    if not os.path.exists(ugalidir):
        from ugali.utils.logger import logger
        msg = "Creating UGALIDIR:\n%s"%ugalidir
        logger.debug(msg)

    return mkdir(ugalidir)

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    (opts, args) = parser.parse_args()

