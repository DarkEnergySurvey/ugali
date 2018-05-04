import os
tag = 'v6'
n_chunk = 100
mc_source_id_start_global = 1
size_batch = 5000 
number_of_batches = 10

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-q','--queue',default='local')
args = parser.parse_args()

# The `csub` command lives in:
# /home/s1/kadrlica/bin/csub
# I try to set it here, but you can always do this in shell
# export PATH=/home/s1/kadrlica/bin:$PATH

os.environ['PATH'] = '/home/s1/kadrlica/bin:'+os.environ['PATH']

logdir = log
if not os.path.exists(logdir): os.makedirs(logdir)

for index_batch in range(0, number_of_batches):
    mc_source_id_start = mc_source_id_start_global + (size_batch * index_batch)
    command = 'python simulate_population.py --tag %s --start %i --size %i --chunk %i'%(tag, mc_source_id_start, size_batch, n_chunk)

    logfile = os.path.join(logdir,'%s.log'%mc_source_id_start)
    submit = 'csub -q %s -o %s '%(args.queue,logfile)+ command
    os.system(command)

    #print command
    #os.system(command)
