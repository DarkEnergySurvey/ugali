#!/usr/bin/env python
"""
Cross-platform batch computing interface.
"""

import subprocess, subprocess as sub
import getpass
from collections import OrderedDict as odict
from itertools import chain
import copy

from ugali.utils.logger import logger

CLUSTERS = odict([
    ('local' ,['local']),
    ('lsf'   ,['lsf','slac','kipac']),
    ('slurm' ,['slurm','midway','kicp']),
    ('condor',['condor','fnal']),
])

QUEUES = odict([
    ('local' ,[]),
    ('lsf'   ,['express','short','medium','long','xlong','xxl',
               'kipac-ibq','bulletmpi']),
    ('slurm' ,[]),
    ('condor',['local','vanilla','universe','grid']),
])

RUNLIMITS = odict([              #Hard limits
        (None       ,'4:00'),    # Default value
        ('express'  ,'0:01'),    # 0:01
        ('short'    ,'0:30'),    # 0:30
        ('medium'   ,'1:00'),    # 4:00
        ('long'     ,'4:00'),    # 32:00
        ('xlong'    ,'72:00'),   # 72:00
        ('xxl'      ,'168:00'),
        ('kipac-ibq','24:00'),   # MPI queues
        ('bulletmpi','72:00'),
        ])

MPIOPTS = odict([
        (None       ,' -R "span[ptile=4]"'),
        ('local'    ,''),
        ('kipac-ibq',' -R "span[ptile=8]"'),
        ('bulletmpi',' -R "span[ptile=16]"'),
        ])

def factory(queue,**kwargs):
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
        raise Exception("Condor cluster not implemented")
    else:
        raise TypeError('Unexpected queue name: %s'%name)

    return batch

batchFactory = factory

class Batch(object):
    # Default options for batch submission
    _defaults = odict([])
    # Map between generic and batch specific names
    _mapping = odict([])

    def __init__(self, **kwargs):
        self.username = getpass.getuser()
        self.default_opts = copy.deepcopy(self._defaults)
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
        logger.debug(command)
        return sub.call(command,shell=True)

    def remap_options(self,opts):
        for k in self._mapping.keys():
            v = opts.pop(k,None)
            if v is not None:
                opts[self._mapping[k]] = v

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
        if opts.get('logfile'): return ' 2>&1 | tee %(logfile)s'%opts
        return ''

class LSF(Batch):
    _defaults = odict([
        ('R','"scratch > 1 && rhel60"'),
        ('C', 0),
        #('M','8G'),
        #('q', 'long'),
    ])

    _mapping = odict([
        ('jobname','J'),
        ('logfile','oo')
    ])

    def __init__(self,**kwargs):
        super(LSF,self).__init__(**kwargs)

        self.jobs_cmd = "bjobs -u %s"%self.username
        self.submit_cmd = "bsub %(opts)s %(command)s"

    def runlimit(self, queue=None):
        """
        Translate queue to wallclock runlimit.
        """
        try:             return RUNLIMITS[queue]
        except KeyError: return RUNLIMITS[None]
    q2w = runlimit

    def mpiopts(self, queue=None):
        """
        Translate queue into MPI options.
        """
        try:             return MPIOPTS[queue]
        except KeyError: return MPIOPTS[None]
    q2mpi = mpiopts

    def parse_options(self, **opts):
        # Default options for the cluster
        options = odict(self.default_opts)
        # Default options for the queue
        #options.update(OPTIONS[options.get('q')])
        # User specified options
        options.update(opts)
        if 'n' in options.keys(): 
            options['a'] = 'mpirun'
            options['R'] += self.mpiopts(options.get('q'))
        options.setdefault('W',self.runlimit(options.get('q')))
        return ''.join('-%s %s '%(k,v) for k,v in options.items())
        
class Slurm(Batch):
    _defaults = odict([
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
    """ Not implemented yet... """
    def __init__(self):
        super(Condor,self).__init__(**kwargs)
        logger.warning('Condor cluster is untested')
        
        self.jobs_cmd = 'condor_q -u %s'%self.username
        self.submit_cmd = "csub %(opts)s %(command)s"

if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()

