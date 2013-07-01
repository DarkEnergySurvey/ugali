"""
Class to construct parabolas from 3 points.
"""

import numpy
import scipy.stats
import scipy.interpolate

############################################################

class Parabola:

    def __init__(self, x, y):
        """
        INPUTS
        x = variable of interest
        y = 2 * log(likelihood)
        """

        # Sort the input
        argsort = numpy.argsort(x)
        self.x = numpy.array(x)[argsort]
        self.y = numpy.array(y)[argsort]

        index = numpy.argmax(self.y)
        if index == 0:
            index_0 = 0
            index_1 = 1
            index_2 = 2
        elif index == len(self.y) - 1:
            index_0 = len(self.y) - 3
            index_1 = len(self.y) - 2
            index_2 = len(self.y) - 1
        else:
            index_0 = index - 1
            index_1 = index
            index_2 = index + 1

        x_0 = self.x[index_0]
        x_1 = self.x[index_1]
        x_2 = self.x[index_2]
        y_0 = self.y[index_0]
        y_1 = self.y[index_1]
        y_2 = self.y[index_2]

        # Invert matrix
        a = numpy.matrix([[x_0**2, x_0, 1.],
                          [x_1**2, x_1, 1.],
                          [x_2**2, x_2, 1.]])
        a_inverse = numpy.linalg.inv(a)
        b = numpy.array([y_0, y_1, y_2])
        p = numpy.dot(numpy.array(a_inverse), b)

        self.p_2 = p[0]
        self.p_1 = p[1]
        self.p_0 = p[2]

        # Vertex
        self.vertex_x = -self.p_1 / (2. * self.p_2)
        self.vertex_y = self.p_0 - (self.p_1**2 / (4. * self.p_2))

    def profileUpperLimit(self, delta = 2.71):
        """
        Compute one-sided upperlimit via profile method.
        """
        a = self.p_2
        b = self.p_1
        if self.vertex_x < 0:
            c = self.p_0 + delta
        else:
            c = delta - self.vertex_y

        if b**2 - 4. * a * c < 0.:
            print 'WARNING'
            return 0.
            
        return max((numpy.sqrt(b**2 - 4. * a * c) - b) / (2. * a), (-1. * numpy.sqrt(b**2 - 4. * a * c) - b) / (2. * a)) 

    def bayesianUpperLimit2(self, alpha, steps = 1.e5):
        """
        Compute one-sided upper limit using Bayesian Method of Helene.
        """
        # Need a check to see whether limit is reliable
        pdf = scipy.interpolate.interp1d(self.x, numpy.exp(self.y / 2.)) # Convert from 2 * log(likelihood) to likelihood
        x_pdf = numpy.linspace(self.x[0], self.x[-1], steps)
        cdf = numpy.cumsum(pdf(x_pdf))
        cdf /= cdf[-1]
        cdf_reflect = scipy.interpolate.interp1d(cdf, x_pdf)
        return cdf_reflect(alpha)
        #return self.x[numpy.argmin((cdf - alpha)**2)]

    def bayesianUpperLimit(self, alpha, steps=1.e5):
        """
        Compute one-sided upper limit using Bayesian Method of Helene.
        """
        #argsort = numpy.argsort(self.x)
        f = scipy.interpolate.interp1d(self.x, self.y, kind='cubic')
        x = numpy.linspace(0., numpy.max(self.x), steps)
        y = numpy.exp(f(x) / 2.)
        cdf = numpy.cumsum(y)
        cdf /= cdf[-1]
        cdf_reflect = scipy.interpolate.interp1d(cdf, x)
        return cdf_reflect(alpha)

        """
        if numpy.isnan(result):
            import pylab

            for ii in range(0, len(self.x)):
                print '%.3f %.3f'%(self.x[ii], self.y[ii])
            
            pylab.figure()
            pylab.scatter(self.x, self.y)
            pylab.figure()
            pylab.scatter(cdf, x)
            raw_input('WAIT')
        
        return result
        """

############################################################

def upperLimitsDeltaTS(confidence_level, one_sided=True, degrees_of_freedom=1):
    """

    """
    if not one_sided:
        confidence_level = 0.5*(confidence_level + 1.)
    ts_min = 0 # TS = Test Statistic
    ts_max = 5
    ts_steps = 1000
    x = numpy.linspace(ts_min, ts_max, ts_steps)
    y = (0.5 * scipy.stats.chi2.sf(x, degrees_of_freedom) - (1. - confidence_level))**2
    return x[numpy.argmin(y)]
        
############################################################
