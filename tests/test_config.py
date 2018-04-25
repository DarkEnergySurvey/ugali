#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import os
import ugali.utils.config

def test_load_config():
    dirname = os.path.dirname(os.path.realpath(ugali.utils.config.__file__))
    configfile = os.path.join(dirname,'..','config/config_y3a2_cel.yaml')
    config = ugali.utils.config.Config(configfile)
    
