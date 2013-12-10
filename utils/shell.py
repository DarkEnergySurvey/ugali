#!/usr/bin/env python
import shutil
import os

# Tools for working with the shell

def pwd():
    # Careful, won't work after a call to os.chdir...
    return os.environ['PWD']

def mkdir(dir):
    if not os.path.exists(dir):  os.makedirs(dir)
    return dir

def mkscratch():
    if os.path.exists('/scratch/'):    
        return(mkdir('/scratch/%s/'%os.environ['USER']))
    elif os.path.exists('/tmp/'):
        return(mkdir('/tmp/%s/'%os.environ['USER']))
    else:
        raise Exception('...')

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    (opts, args) = parser.parse_args()

