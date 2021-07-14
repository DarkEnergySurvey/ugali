"""
Class to construct parabolas from 3 points.

ADW: Need to get rid of all of the plotting stuff
ADW: Doesn't this all exist in np.poly?
"""

import numpy as np
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
        argsort = np.argsort(x)
        self.x = np.array(x)[argsort]
        self.y = np.array(y)[argsort]

        index = np.argmax(self.y)
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
        a = np.matrix([[x_0**2, x_0, 1.],
                          [x_1**2, x_1, 1.],
                          [x_2**2, x_2, 1.]])
        a_inverse = np.linalg.inv(a)
        b = np.array([y_0, y_1, y_2])
        p = np.dot(np.array(a_inverse), b)

        self.p_2 = p[0]
        self.p_1 = p[1]
        self.p_0 = p[2]
         
        # Vertex
        self.vertex_x = -self.p_1 / (2. * self.p_2)
        self.vertex_y = self.p_0 - (self.p_1**2 / (4. * self.p_2))

    def __eq__(self,other):
        return np.allclose([self.p_0,self.p_1,self.p_2],[other.p_0,other.p_1,other.p_2])

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
            x.append(np.linspace(self.x[ii], self.x[ii + 1], factor)[0: -1])
            y.append(p(x[-1]))

        p = Parabola(self.x[len(self.x) - 3:], self.y[len(self.y) - 3:])
        x.append(np.linspace(self.x[-2], self.x[-1], factor)[0: -1])
        y.append(p(x[-1]))

        x.append([self.x[-1]])
        y.append([self.y[-1]])

        #f = scipy.interpolate.interp1d(np.concatenate(x), np.concatenate(y))
        #x = np.linspace(self.x[0], self.x[-1], len(x) * factor)   
        #return x, f(x)
        
        return np.concatenate(x), np.concatenate(y)

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
            print('WARNING')
            print(a, b, c)
            return 0.
            
        return max((np.sqrt(b**2 - 4. * a * c) - b) / (2. * a), (-1. * np.sqrt(b**2 - 4. * a * c) - b) / (2. * a)) 

    def bayesianUpperLimit(self, alpha, steps=1e5, plot=False):
        """
        Compute one-sided upper limit using Bayesian Method of Helene.
        Several methods of increasing numerical stability have been implemented.
        """
        x_dense, y_dense = self.densify()
        y_dense -= np.max(y_dense) # Numeric stability
        f = scipy.interpolate.interp1d(x_dense, y_dense, kind='linear')
        x = np.linspace(0., np.max(x_dense), int(steps))
        pdf = np.exp(f(x) / 2.)
        cut = (pdf / np.max(pdf)) > 1.e-10
        x = x[cut]
        pdf = pdf[cut]
        #pdf /= pdf[0]
        #forbidden = np.nonzero(pdf < 1.e-10)[0]
        #if len(forbidden) > 0:
        #    index = forbidden[0] # Numeric stability
        #    x = x[0: index]
        #    pdf = pdf[0: index]
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]
        cdf_reflect = scipy.interpolate.interp1d(cdf, x)

        return cdf_reflect(alpha)

    def bayesianUpperLimit2(self, alpha, steps=1e5, plot=False):
        """
        Compute one-sided upper limit using Bayesian Method of Helene.
        """
        cut = ((self.y / 2.) > -30.) # Numeric stability
        try:
            f = scipy.interpolate.interp1d(self.x[cut], self.y[cut], kind='cubic')
        except:
            f = scipy.interpolate.interp1d(self.x[cut], self.y[cut], kind='linear')
        x = np.linspace(0., np.max(self.x[cut]), int(steps))
        y = np.exp(f(x) / 2.)
        #forbidden = np.nonzero((y / np.exp(self.vertex_y / 2.)) < 1.e-10)[0]
        forbidden = np.nonzero((y / self.vertex_y) < 1.e-10)[0]
        if len(forbidden) > 0:
            index = forbidden[0] # Numeric stability
            x = x[0: index]
            y = y[0: index]
        cdf = np.cumsum(y)
        cdf /= cdf[-1]
        cdf_reflect = scipy.interpolate.interp1d(cdf, x)

        return cdf_reflect(alpha)


    def confidenceInterval(self, alpha=0.6827, steps=1e5, plot=False):
        """
        Compute two-sided confidence interval by taking x-values corresponding to the largest PDF-values first.
        """
        x_dense, y_dense = self.densify()
        y_dense -= np.max(y_dense) # Numeric stability
        f = scipy.interpolate.interp1d(x_dense, y_dense, kind='linear')
        x = np.linspace(0., np.max(x_dense), int(steps))
        # ADW: Why does this start at 0, which often outside the input range?
        # Wouldn't starting at xmin be better:
        #x = np.linspace(np.min(x_dense), np.max(x_dense), int(steps))
        pdf = np.exp(f(x) / 2.)
        cut = (pdf / np.max(pdf)) > 1.e-10
        x = x[cut]
        pdf = pdf[cut]

        sorted_pdf_indices = np.argsort(pdf)[::-1] # Indices of PDF in descending value
        cdf = np.cumsum(pdf[sorted_pdf_indices])
        cdf /= cdf[-1]
        sorted_pdf_index_max = np.argmin((cdf - alpha)**2)
        x_select = x[sorted_pdf_indices[0: sorted_pdf_index_max]]
            
        return np.min(x_select), np.max(x_select) 

############################################################

def upperLimitsDeltaTS(confidence_level, one_sided=True, degrees_of_freedom=1):
    """

    """
    if not one_sided:
        confidence_level = 0.5*(confidence_level + 1.)
    ts_min = 0 # TS = Test Statistic
    ts_max = 5
    ts_steps = 1000
    x = np.linspace(ts_min, ts_max, int(ts_steps))
    y = (0.5 * scipy.stats.chi2.sf(x, degrees_of_freedom) - (1. - confidence_level))**2
    return x[np.argmin(y)]
        
############################################################
