"""
Class to construct parabolas from 3 points.

ADW: Need to move all of the plotting stuff
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

    def __eq__(self,other):
        return numpy.allclose([self.p_0,self.p_1,self.p_2],[other.p_0,other.p_1,other.p_2])

    def __ne__(self,other):
        return not self.__eq__(other)
        
    def __repr__(self):
        return "y = %.2g * x**2 + %.2g * x + %.2g"%(self.p_2, self.p_1, self.p_0)

    def __str__(self):
        return self.__repr__()

    def __call__(self, x):
        """
        Evaluate the parabola.
        """
        return (self.p_2 * x**2) + (self.p_1 * x) + self.p_0

    def densify(self, factor=10):
        """
        Increase the density of points along the parabolic curve.
        """
        x = []
        y = []
        for ii in range(0, len(self.x) - 2):
            p = Parabola(self.x[ii: ii + 3], self.y[ii: ii + 3])
            x.append(numpy.linspace(self.x[ii], self.x[ii + 1], factor)[0: -1])
            y.append(p(x[-1]))

        p = Parabola(self.x[len(self.x) - 3:], self.y[len(self.y) - 3:])
        x.append(numpy.linspace(self.x[-2], self.x[-1], factor)[0: -1])
        y.append(p(x[-1]))

        x.append([self.x[-1]])
        y.append([self.y[-1]])

        #f = scipy.interpolate.interp1d(numpy.concatenate(x), numpy.concatenate(y))
        #x = numpy.linspace(self.x[0], self.x[-1], len(x) * factor)   
        #return x, f(x)
        
        return numpy.concatenate(x), numpy.concatenate(y)

    def profileUpperLimit(self, delta = 2.71):
        """
        Compute one-sided upperlimit via profile method.
        """
        a = self.p_2
        b = self.p_1
        if self.vertex_x < 0:
            c = self.p_0 + delta
        else:
            c = self.p_0 - self.vertex_y + delta

        if b**2 - 4. * a * c < 0.:
            print 'WARNING'

            print a, b, c
            
            #pylab.figure()
            #pylab.scatter(self.x, self.y)
            #raw_input('WAIT')
            return 0.

        
            
        return max((numpy.sqrt(b**2 - 4. * a * c) - b) / (2. * a), (-1. * numpy.sqrt(b**2 - 4. * a * c) - b) / (2. * a)) 

    #def bayesianUpperLimit3(self, alpha, steps = 1.e5):
    #    """
    #    Compute one-sided upper limit using Bayesian Method of Helene.
    #    """
    #    # Need a check to see whether limit is reliable
    #    pdf = scipy.interpolate.interp1d(self.x, numpy.exp(self.y / 2.)) # Convert from 2 * log(likelihood) to likelihood
    #    x_pdf = numpy.linspace(self.x[0], self.x[-1], steps)
    #    cdf = numpy.cumsum(pdf(x_pdf))
    #    cdf /= cdf[-1]
    #    cdf_reflect = scipy.interpolate.interp1d(cdf, x_pdf)
    #    return cdf_reflect(alpha)
    #    #return self.x[numpy.argmin((cdf - alpha)**2)]

    def bayesianUpperLimit(self, alpha, steps=1.e5, plot=False):
        """
        Compute one-sided upper limit using Bayesian Method of Helene.
        Several methods of increasing numerical stability have been implemented.
        """
        x_dense, y_dense = self.densify()
        y_dense -= numpy.max(y_dense) # Numeric stability
        f = scipy.interpolate.interp1d(x_dense, y_dense, kind='linear')
        x = numpy.linspace(0., numpy.max(x_dense), steps)
        pdf = numpy.exp(f(x) / 2.)
        cut = (pdf / numpy.max(pdf)) > 1.e-10
        x = x[cut]
        pdf = pdf[cut]
        #pdf /= pdf[0]
        #forbidden = numpy.nonzero(pdf < 1.e-10)[0]
        #if len(forbidden) > 0:
        #    index = forbidden[0] # Numeric stability
        #    x = x[0: index]
        #    pdf = pdf[0: index]
        cdf = numpy.cumsum(pdf)
        cdf /= cdf[-1]
        cdf_reflect = scipy.interpolate.interp1d(cdf, x)

        #if plot:            
        #    pylab.figure()
        #    pylab.plot(x, f(x))
        #    pylab.scatter(self.x, self.y, c='red')
        #    
        #    pylab.figure()
        #    pylab.plot(x, pdf)
        #    
        #    pylab.figure()
        #    pylab.plot(cdf, x)
        
        return cdf_reflect(alpha)

    def bayesianUpperLimit2(self, alpha, steps=1.e5, plot=False):
        """
        Compute one-sided upper limit using Bayesian Method of Helene.
        """
        cut = ((self.y / 2.) > -30.) # Numeric stability
        try:
            f = scipy.interpolate.interp1d(self.x[cut], self.y[cut], kind='cubic')
        except:
            f = scipy.interpolate.interp1d(self.x[cut], self.y[cut], kind='linear')
        x = numpy.linspace(0., numpy.max(self.x[cut]), steps)
        y = numpy.exp(f(x) / 2.)
        #forbidden = numpy.nonzero((y / numpy.exp(self.vertex_y / 2.)) < 1.e-10)[0]
        forbidden = numpy.nonzero((y / self.vertex_y) < 1.e-10)[0]
        if len(forbidden) > 0:
            index = forbidden[0] # Numeric stability
            x = x[0: index]
            y = y[0: index]
        cdf = numpy.cumsum(y)
        cdf /= cdf[-1]
        cdf_reflect = scipy.interpolate.interp1d(cdf, x)

        #if plot:
        #    pylab.figure()
        #    pylab.scatter(self.x, self.y)
        # 
        #    pylab.figure()
        #    pylab.plot(x, f(x))
        #    
        #    pylab.figure()
        #    pylab.plot(x, y)
        #    
        #    pylab.figure()
        #    pylab.plot(cdf, x)
        
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

    def confidenceInterval(self, alpha=0.6827, steps=1.e5, plot=False):
        """
        Compute two-sided confidence interval by taking x-values corresponding to the largest PDF-values first.
        """
        x_dense, y_dense = self.densify()
        y_dense -= numpy.max(y_dense) # Numeric stability
        f = scipy.interpolate.interp1d(x_dense, y_dense, kind='linear')
        x = numpy.linspace(0., numpy.max(x_dense), steps)
        # ADW: Why does this start at 0, which often outside the input range?
        # Wouldn't starting at xmin be better:
        #x = numpy.linspace(numpy.min(x_dense), numpy.max(x_dense), steps)
        pdf = numpy.exp(f(x) / 2.)
        cut = (pdf / numpy.max(pdf)) > 1.e-10
        x = x[cut]
        pdf = pdf[cut]

        sorted_pdf_indices = numpy.argsort(pdf)[::-1] # Indices of PDF in descending value
        cdf = numpy.cumsum(pdf[sorted_pdf_indices])
        cdf /= cdf[-1]
        sorted_pdf_index_max = numpy.argmin((cdf - alpha)**2)
        x_select = x[sorted_pdf_indices[0: sorted_pdf_index_max]]

        #if plot:
        #    cdf = numpy.cumsum(pdf)
        #    cdf /= cdf[-1]
        #    print cdf[numpy.max(sorted_pdf_indices[0: sorted_pdf_index_max])] \
        #          - cdf[numpy.min(sorted_pdf_indices[0: sorted_pdf_index_max])]
        #    
        #    pylab.figure()
        #    pylab.plot(x, f(x))
        #    pylab.scatter(self.x, self.y, c='red')
        #    
        #    pylab.figure()
        #    pylab.plot(x, pdf)
            
        return numpy.min(x_select), numpy.max(x_select) 

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
