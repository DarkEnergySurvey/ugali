import os
import subprocess
import numpy as np

tag = 'v7'
n_chunk = 100
mc_source_id_start_global = 1
size_batch = 1000 
number_of_batches = 10

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-q','--queue',default='local')
parser.add_argument('-t','--tag',default=tag)
parser.add_argument('-k','--nchunk',default=n_chunk,type=int)
parser.add_argument('-b','--size-batch',default=size_batch,type=int)
parser.add_argument('-n','--nbatch',default=number_of_batches,type=int)
parser.add_argument('--mc_source_id',default=mc_source_id_start_global,type=int)

args = parser.parse_args()

# The `csub` command lives in:
# /home/s1/kadrlica/bin/csub
# I try to set it here, but you can always do this in shell
# export PATH=/home/s1/kadrlica/bin:$PATH

os.environ['PATH'] = '/home/s1/kadrlica/bin:'+os.environ['PATH']

logdir = '%s/log'%args.tag
if not os.path.exists(logdir): os.makedirs(logdir)

for index_batch in np.arange(args.nbatch):
    mc_source_id_start = args.mc_source_id + (args.size_batch * index_batch)
    command = 'simulate_population.py --tag %s --start %i --size %i --chunk %i'%(args.tag, mc_source_id_start, args.size_batch, args.nchunk)

    logfile = os.path.join(logdir,'%s.log'%mc_source_id_start)
    submit = 'csub -q %s -o %s -n 20 '%(args.queue,logfile) + command
    subprocess.call(submit,shell=True)

    #print command
    #os.system(command)
