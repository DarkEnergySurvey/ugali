#!/usr/bin/env python
import sys

import numpy as np
import pylab as plt

from ugali.analysis.pipeline import Pipeline
import ugali.utils.plotting

description="Plot distance modulus panels"
components = []

def run(self):
    coords = self.opts.coords
    names = vars(self.opts).get('names',len(coords)*[''])
    labels=[n.lower().replace(' ','_').replace('(','').replace(')','') for n in names]

    for name,label,coord in zip(names,labels,coords):
        glon,glat = coord[0],coord[1]
        print('\t',name,'(%.2f,%.2f)'%(glon,glat))
        plotter =ugali.utils.plotting.BasePlotter(glon,glat,self.config,radius=0.5)
        plotter.plotDistance()
        outfile='%s_distances.png'%label
        plt.savefig(outfile,bbox_inches='tight')

Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parser.add_coords(radius=True,targets=True,required=True)
pipeline.parse_args()
pipeline.execute()


