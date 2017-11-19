#!/usr/bin/env python
"""
Compile candidate list from results_dir
"""
__author__ = "Sidney Mau"

import glob
import yaml

with open('config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

candidate_list = cfg['candidate_list']


results_file = open(candidate_list, 'w')
results_file.write('sig, ra, dec, distance_modulus, r\n')
for file in glob.glob('{}/*.csv'.format(cfg['results_dir'])):
    writer = open(file, 'r')
    for line in writer:
        results_file.write(line)
    writer.close()
results_file.close()
