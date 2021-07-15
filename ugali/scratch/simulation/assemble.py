DeprecationWarning("'assemble.py' should be removed")

import sys
import os
import shutil
import yaml

config_file = 'des_y3a2_survey_selection_function.yaml'
outdir = 'des_y3a2_survey_selection_function_v5_test'

config = yaml.load(open(config_file))

if os.path.exists(outdir):
    print("Output directory exists: %s"%outdir)
    input('Are you sure you want to continue? [Press ENTER to continue]')

if not os.path.exists(outdir):
    os.mkdir(outdir)

shutil.copy(config['infile']['fracdet'], outdir + '/.')
shutil.copy(config['infile']['population_metadata'], outdir + '/.')
shutil.copy(config['simple']['real_results'], outdir + '/.')
shutil.copy(config['simple']['sim_results'], outdir + '/.')
shutil.copy(config['simple']['classifier'], outdir + '/.')
shutil.copy('survey_selection_function.py', outdir + '/.')

config['infile']['fracdet'] = os.path.basename(config['infile']['fracdet'])
config['infile']['population_metadata'] = os.path.basename(config['infile']['population_metadata'])
config['simple']['real_results'] = os.path.basename(config['simple']['real_results'])
config['simple']['sim_results'] = os.path.basename(config['simple']['sim_results'])
config['simple']['classifier'] = os.path.basename(config['simple']['classifier'])

writer = open('%s/%s'%(outdir, config_file), 'w')
writer.write(yaml.dump(config, default_flow_style=False))
writer.close()

os.system('tar -zcvf %s.tar.gz %s'%(outdir, outdir))
