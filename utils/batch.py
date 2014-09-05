#!/usr/bin/env python
import subprocess, subprocess as sub
import getpass
from collections import OrderedDict as odict
from itertools import chain

from ugali.utils.logger import logger

CLUSTERS = odict([
    ('local',['local']),
    ('lsf',['lsf','slac','kipac']),
    ('slurm',['slurm','midway','kicp']),
    ('condor',['condor','fnal']),
])

QUEUES = odict([
    ('local',[]),
    ('lsf',['express','short','medium','long','xlong','xxl','kipac-ibq']),
    ('slurm',[]),
    ('condor',['vanilla','universe']),
])

def batchFactory(queue,**kwargs):
    if queue is None: queue = 'local'

    name = queue.lower()
    if name in list(chain(*QUEUES.values())):
        kwargs.setdefault('q',name)

    if name in CLUSTERS['local']+QUEUES['local']:
        batch = Local(**kwargs)
    elif name in CLUSTERS['lsf']+QUEUES['lsf']:
        batch = LSF(**kwargs)
    elif name in CLUSTERS['slurm']+QUEUES['slurm']:
        batch = Slurm(**kwargs)
    elif name in CLUSTERS['condor']+QUEUES['condor']:
        # Need to learn how to use condor first...
        batch = Condor(**kwargs)
        raise Exception("FNAL cluster not implemented")
    else:
        raise TypeError('Unexpected queue name: %s'%name)

    return batch

class Batch(object):
    # Default options for batch submission
    default_opts = odict([])
    # Map between generic and batch specific names
    map_opts = odict([])

    def __init__(self, **kwargs):
        self.username = getpass.getuser()
        self.default_opts.update(**kwargs)
        self.submit_cmd = "submit %(opts)s %(command)s"
        self.jobs_cmd = "jobs"

    def jobs(self):
        out = self.popen(self.jobs_cmd)
        stdout = out.communicate()[0]
        return stdout

    def njobs(self):
        # Remove header line
        jobs = self.jobs()
        return len(jobs.strip().split('\n'))-1 if jobs else 0

    def popen(self, command):
        return sub.Popen(command,shell=True,
                         stdin=sub.PIPE,stdout=sub.PIPE,stderr=sub.PIPE)

    def call(self, command):
        return sub.call(command,shell=True)

    def remap_options(self,opts):
        for k in self.map_opts.keys():
            v = opts.pop(k,None)
            if v is not None:
                opts[self.map_opts[k]] = v

    def batch(self, command, jobname=None, logfile=None, **opts):
        if jobname: opts.update(jobname=jobname)
        if logfile: opts.update(logfile=logfile)
        self.remap_options(opts)
        params = dict(opts=self.parse_options(**opts),command=command)
        return self.submit_cmd%params

    def submit(self, command, jobname=None, logfile=None, **opts):
        cmd = self.batch(command, jobname, logfile, **opts)
        self.call(cmd)
        return cmd

class Local(Batch):
    def __init__(self,**kwargs):
        super(Local,self).__init__(**kwargs)
        self.jobs_cmd = "echo 0"
        self.submit_cmd = "%(command)s %(opts)s"

    def parse_options(self,**opts):
        if opts.get('logfile'): return ' | tee %(logfile)s'%opts
        return ''

class LSF(Batch):
    default_opts = odict([
        ('R','"scratch > 1 && rhel60"'),
        ('C', 0),
        ('q', 'long'),
    ])

    map_opts = odict([
        ('jobname','J'),
        ('logfile','oo')
    ])

    def __init__(self,**kwargs):
        super(LSF,self).__init__(**kwargs)

        self.jobs_cmd = "bjobs -u %s"%self.username
        self.submit_cmd = "bsub %(opts)s %(command)s"
    
    def parse_options(self, **opts):
        options = odict(self.default_opts)
        options.update(opts)
        return ''.join('-%s %s '%(k,v) for k,v in options.items())
        
class Slurm(Batch):
    default_opts = odict([
        ('account','kicp'),
        ('partition','kicp-ht'),
        ('mem',10000)
    ])

    def __init__(self, **kwargs):
        super(Slurm,self).__init__(**kwargs)
        logger.warning('Slurm cluster is untested')

        self.jobs_cmd = "squeue -u %s"%self.username
        self.submit_cmd = "sbatch %(opts)s %(command)s"

    def parse_options(self, **opts):
        options = odict(self.default_opts)
        options.update(opts)
        return ''.join('--%s %s '%(k,v) for k,v in options.items())

class Condor(Batch):
    default_opts = odict()
    map_opts = odict()

    def __init__(self):
        super(Condor,self).__init__(**kwargs)
        logger.warning('Condor cluster is untested')
        
        self.jobs_cmd = 'condor_q -u %s'%self.username
        self.submit_cmd = "csub %(opts)s %(command)s"

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('args',nargs=argparse.REMAINDER)
    opts = parser.parse_args(); args = opts.args
