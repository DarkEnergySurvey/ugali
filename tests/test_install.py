#!/usr/bin/env python
"""
Test installation
"""
import os
import subprocess
import tempfile

GITURL = "https://github.com/DarkEnergySurvey/ugali"
GITURL = "https://github.com/kadrlica/ugali"

def call_cmd(cmd):
    print(cmd)
    subprocess.check_call(cmd,shell=True)

def call_chdir(path):
    if not os.path.isdir(path):
        raise Exception("Directory does not exist: %s"%path)
    print('os.chdir(%s)'%path)
    os.chdir(path)

def setup_env():
    path='./lib/python2.7/site-packages/'
    call_cmd("mkdir -p %s"%path)
    env = 'HOME=.; PYTHONPATH=$PYTHONPATH:%s;'%path
    return path, env

def call_setup_py():
    path,env = setup_env()
    #cmd = env +' python setup.py -q install -f --prefix=.'
    #cmd = env +' python setup.py -v install -f --user'
    call_cmd(cmd)
    #call_cmd(cmd + ' --isochrones')
    #call_cmd(cmd + ' --isochrones --isochrones-path=./tmp')

def call_pip():
    path,env = setup_env()
    cmd = env + 'pip install --user --no-deps -I  ugali'
    call_cmd(cmd)
    #call_cmd(cmd +' --install-option "--isochrones"')
    #call_cmd(cmd +' --install-option "--isochrones" --install-option="--isochrones-path=./tmp" ')
    #call_cmd('pip uninstall ugali')

def test_git_install():
    cwd = os.getcwd()
    tempdir = tempfile.mkdtemp()
    call_chdir(tempdir)
    call_cmd('git clone %s.git'%GITURL)
    call_chdir('ugali')
    call_setup_py()
    call_cmd('rm -rf %s'%tempdir)
    call_chdir(cwd)

#def test_zip_install():
#    tempdir = tempfile.mkdtemp()
#    call_chdir(tempdir)
#    call_cmd('wget %s/archive/master.zip'%GITURL)
#    call_cmd('unzip -q master.zip')
#    call_chdir('ugali-master')
#    call_setup_py()
#    call_cmd('rm -rf %s'%tempdir)
#    
#def test_pip_install():
#    tempdir = tempfile.mkdtemp()
#    call_chdir(tempdir)
#    call_pip()
#    call_cmd('rm -rf %s'%tempdir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    #test_git_install()
    #test_zip_install()
    #test_pip_install()
