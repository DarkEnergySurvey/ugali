#!/usr/bin/env python
"""
Testing the pipeline (probably a bit outside the scope of unit tests).
"""
__author__ = "Alex Drlica-Wagner"
import glob
import os
import subprocess

def _test_pipeline(n=2,args=None):
    pipedir='./ugali/pipeline'
    
    script = glob.glob(os.path.join(pipedir,'run_%02d.0*.py'%n))[0]
    cmd = script + ' ./ugali/config/config_test.yaml'
    if args: cmd += ' '+args
    subprocess.check_call(cmd,shell=True)

#def test_pipeline_02():
#    _test_pipeline(2)

#def test_pipeline_03():
#    _test_pipeline(3,'-f -q local --cel 53.92 -54.05')

#def test_pipeline_04():
#    _test_pipeline(4)
 
#def test_pipeline_05():
#    _test_pipeline(5,'-q local --name reticulum_ii --srcmdl -r mcmc')

if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()

    #test_pipeline_03()
    #test_pipeline_05()
