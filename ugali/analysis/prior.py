#!/usr/bin/env python

class Prior(object):
    def __call__(self, value):
        return self.value(value)
 
    def value(self, value):
        msg = "`value` must be implmented in child class"
        raise Exception(msg)

class UniformPrior(Prior): 
    def value(self, value):
        return 1.0

class InversePrior(Prior):
    def value(self, value):
        return 1.0/value

class BetaPrior(Prior):
    def value(self, value):
        return scipy.stats.beta.pdf(value,0.5,0.5)

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    opts = parser.parse_args()

