import os
tag = 'v6'
n_chunk = 100
mc_source_id_start_global = 1
size_batch = 5000 
number_of_batches = 10

for index_batch in range(0, number_of_batches):
    mc_source_id_start = mc_source_id_start_global + (size_batch * index_batch)
    command = 'python simulate_population.py --tag %s --start %i --size %i --chunk %i'%(tag, mc_source_id_start, size_batch, n_chunk)
    #print command
    os.system(command)
