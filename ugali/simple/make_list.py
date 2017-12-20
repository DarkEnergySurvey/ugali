#!/usr/bin/env python
"""
Compile candidate list from results_dir
"""
__author__ = "Sidney Mau"

import glob
import yaml

with open('config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

candidate_list = cfg[cfg['data']]['candidate_list']


results_file = open(candidate_list, 'w')
results_file.write('SIG, RA, DEC, MODULUS, r\n')
for file in glob.glob('{}/*.csv'.format(cfg[cfg['data']]['results_dir'])):
    writer = open(file, 'r')
    for line in writer:
        results_file.write(line)
    writer.close()
results_file.close()
