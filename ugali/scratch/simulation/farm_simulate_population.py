#!/usr/bin/env python
import os
import subprocess

# ADW: It'd be better to track this in the config
tag = 'ps1_v1' # PS1
tag = 'des_v7' # DES
tag = 'dc2_v1' # LSST DC2
tag = 'lsst_dc2_v2' # LSST DP0 (DC2)
tag = 'lsst_dp0_v1' # LSST DP0 (DC2)
n_chunk = 100
mc_source_id_start_global = 1
size_batch = 1000 
number_of_batches = 10

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('config')
parser.add_argument('-q','--queue',default='local')
parser.add_argument('-t','--tag',default=tag,
                    help='tag appended to file name')
parser.add_argument('-k','--nchunk',default=n_chunk,type=int,
                    help='number of satellites per catalog file')
parser.add_argument('-b','--size-batch',default=size_batch,type=int,
                    help='number of satellites per population file')
parser.add_argument('-n','--nbatch',default=number_of_batches,type=int,
                    help='number of batches to submit')
parser.add_argument('--mc_source_id',default=mc_source_id_start_global,type=int,
                    help='unique identifier')
parser.add_argument('-s','--section',default='des',
                    choices=['des','ps1','lsst_dc2','lsst_dp0'],
                    help='section of config file')
parser.add_argument('--dwarfs', dest='dwarfs', action='store_true', 
                    help="Simulate from known dwarfs")
parser.add_argument('--sleep',default=None,type=int,
                    help='sleep between jobs (seconds)')
parser.add_argument('--njobs',default=10,type=int,
                    help='number of jobs to run concurrently')
parser.add_argument('--host',default=None,metavar='host1[,host2,...]',
                    help="submit to comma-delimited list of hosts or 'all'")

args = parser.parse_args()

# The `csub` command lives in:
# /home/s1/kadrlica/bin/csub
# I try to set it here, but you can always do this in shell
# export PATH=/home/s1/kadrlica/bin:$PATH
os.environ['PATH'] = '/home/s1/kadrlica/bin:'+os.environ['PATH']

logdir = '%s/log'%args.tag
if not os.path.exists(logdir): os.makedirs(logdir)

for index_batch in range(args.nbatch):
    seed = mc_source_id_start = args.mc_source_id + (args.size_batch * index_batch)
    dwarfs = '--dwarfs' if args.dwarfs else ''
    command = 'simulate_population.py %s -s %s --tag %s --start %i --size %i --chunk %i --seed %i %s'%(args.config, args.section, args.tag, mc_source_id_start, args.size_batch, args.nchunk, seed, dwarfs)
    
    # Max number of jobs limited by memory
    logfile = os.path.join(logdir,'%07i.log'%mc_source_id_start)
    params = dict(queue=args.queue,logfile=logfile,
                  sleep='-s %d'%args.sleep if args.sleep else '',
                  host='--host %s'%args.host if args.host else '',
                  njobs='-n %s'%args.njobs if args.njobs else '',
                  command=command)
    submit = 'csub -q %(queue)s %(njobs)s %(sleep)s %(host)s -o %(logfile)s %(command)s'%params
    print(submit)
    subprocess.call(submit,shell=True)

