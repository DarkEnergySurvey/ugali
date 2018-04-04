#!/usr/bin/env python
from os.path import splitext,basename

import pylab as plt
import numpy as np


if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    (opts, args) = parser.parse_args()


    for arg in args:
        if splitext(arg)[1] != '.dat':
           raise Exception('Input not .dat files')
        params = basename(arg).split('_')
        label = params[0]
        ext = params[4]

        s = ''.join(open(arg).readlines())
        results = eval(s)
        logLike = np.array(results['log_likelihood'])
        low,rich,up = np.array([results['richness_lower'],
                                results['richness'],
                                results['richness_upper']])

        mass = np.array(results['stellar_mass'])

        norm = mass/rich
        low *= norm
        rich *= norm
        up *= norm
        plt.errorbar(list(range(len(rich))), rich, yerr=[rich-low, up-rich], fmt='o',label=label)
        plt.axhline(np.mean(rich),ls='--',color='r')

        for mc_mass in np.unique(results['mc_stellar_mass']):
            plt.axhline(mc_mass,ls='--',color='k')

    plt.title(r'Likelihood Comparison ($r_h = %s$ deg)'%ext)
    plt.ylabel(r'Stellar Mass ($M_{\odot}$)')
    plt.legend(loc='upper right')
    plt.savefig("%s_%s.png"%(params[1],ext))
