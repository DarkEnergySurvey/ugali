#!/usr/bin/env python
"""
Test the installation process.
"""
import os
from os.path import basename, dirname
import subprocess
import tempfile

# ADW: Is there any reason to do this in CI?
__test__ = False

# Not all that robust...
try:
    GITBASE = basename(dirname(subprocess.check_output('git config --get remote.origin.url',shell=True).split(':')[1]))
except:
    GITBASE = 'DarkEnergySurvey'

GITURL = "https://github.com/%s/ugali"%GITBASE

def call_cmd(cmd):
    print(cmd)
    subprocess.check_call(cmd,shell=True)

def call_chdir(path):
    if not os.path.isdir(path):
        raise Exception("Directory does not exist: %s"%path)
    print('os.chdir(%s)'%path)
    os.chdir(path)

def setup_env():
    cwd = os.getcwd()
    path='%s/lib/python2.7/site-packages/'%cwd
    env = 'CWD=%s; PYTHONPATH=$PYTHONPATH:%s; '%(cwd,path)
    call_cmd("mkdir -p %s"%path)
    return path, env

def call_setup_py():
    path,env = setup_env()
    cmd = env + 'python setup.py -q install -f --prefix=$CWD'
    call_cmd(cmd)
    call_cmd(cmd + ' --isochrones')
    call_cmd(cmd + ' --isochrones --isochrones-path=$CWD/tmp')

def call_pip():
    path,env = setup_env()
    # This should work, but doesn't
    #cmd = env + 'pip install --no-deps -I -t $CWD ugali'
    cmd = env + 'pip install --no-deps -I ugali --install-option "--prefix=$CWD"'
    call_cmd(cmd)
    call_cmd(cmd +' --install-option "--isochrones"')
    call_cmd(cmd +' --install-option "--isochrones" --install-option="--isochrones-path=$CWD/tmp" ')

def test_git_install():
    tempdir = tempfile.mkdtemp()
    call_chdir(tempdir)
    call_cmd('git clone %s.git'%GITURL)
    call_chdir('ugali')
    call_setup_py()
    call_chdir(os.path.expandvars('$HOME'))
    call_cmd('rm -rf %s'%tempdir)
 
def test_zip_install():
    tempdir = tempfile.mkdtemp()
    call_chdir(tempdir)
    call_cmd('wget %s/archive/master.zip'%GITURL)
    call_cmd('unzip -q master.zip')
    call_chdir('ugali-master')
    call_setup_py()
    call_chdir(os.path.expandvars('$HOME'))
    call_cmd('rm -rf %s'%tempdir)

def test_pip_install():
    tempdir = tempfile.mkdtemp()
    call_chdir(tempdir)
    call_pip()
    call_chdir(os.path.expandvars('$HOME'))
    call_cmd('rm -rf %s'%tempdir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    test_git_install()
    test_zip_install()
    test_pip_install()
