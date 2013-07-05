"""
Class for converting between sphere to image coordinates using map projections.

Based on Calabretta & Greisen 2002, A&A, 357, 1077-1122
"""

import numpy
import healpy

############################################################

class SphericalRotator:
    """
    Base class for rotating points on a sphere.

    The input is a fiducial point (deg) which becomes (0, 0) in rotated coordinates.
    """

    def __init__(self, lon_ref, lat_ref, zenithal=False):
        self.setReference(lon_ref, lat_ref, zenithal)

    def setReference(self, lon_ref, lat_ref, zenithal=False):

        if zenithal:
            phi = (numpy.pi / 2.) + numpy.radians(lon_ref)
            theta = (numpy.pi / 2.) - numpy.radians(lat_ref)
            psi = 0.
        if not zenithal:
            phi = (-numpy.pi / 2.) + numpy.radians(lon_ref)
            theta = numpy.radians(lat_ref)
            psi = numpy.radians(90.) # psi = 90 corresponds to (0, 0) psi = -90 corresponds to (180, 0)
        
        self.rotation_matrix = numpy.matrix([[numpy.cos(psi) * numpy.cos(phi) - numpy.cos(theta) * numpy.sin(phi) * numpy.sin(psi),
                                              numpy.cos(psi) * numpy.sin(phi) + numpy.cos(theta) * numpy.cos(phi) * numpy.sin(psi),
                                              numpy.sin(psi) * numpy.sin(theta)],
                                             [-numpy.sin(psi) * numpy.cos(phi) - numpy.cos(theta) * numpy.sin(phi) * numpy.cos(psi),
                                              -numpy.sin(psi) * numpy.sin(phi) + numpy.cos(theta) * numpy.cos(phi) * numpy.cos(psi),
                                              numpy.cos(psi)*numpy.sin(theta)],
                                             [numpy.sin(theta) * numpy.sin(phi),
                                              -numpy.sin(theta) * numpy.cos(phi),
                                              numpy.cos(theta)]])
        
        self.inverted_rotation_matrix = numpy.linalg.inv(self.rotation_matrix)

    def rotate(self, lon, lat, invert = False):
        lon = numpy.radians(lon)
        lat = numpy.radians(lat) 

        vec = numpy.array([numpy.cos(lat) * numpy.cos(lon),
                           numpy.cos(lat) * numpy.sin(lon),
                           numpy.sin(lat)])

        if invert:
            vec_prime = numpy.dot(numpy.array(self.inverted_rotation_matrix), vec)
        else:        
            vec_prime = numpy.dot(numpy.array(self.rotation_matrix), vec)

        lon_prime = numpy.arctan2(vec_prime[1], vec_prime[0])
        lat_prime = numpy.arcsin(vec_prime[2])

        return (numpy.degrees(lon_prime) % 360.), numpy.degrees(lat_prime)

############################################################

class Projector:
    """
    Class for performing two-dimensional map projections from the celestial sphere.
    """

    def __init__(self, lon_ref, lat_ref, proj_type = 'ait'):
        self.lon_ref = lon_ref
        self.lat_ref = lat_ref
        self.proj_type = proj_type

        if proj_type.lower() == 'ait':
            self.rotator = SphericalRotator(lon_ref, lat_ref, zenithal=False)
            self.sphere_to_image_func = aitoffSphereToImage
            self.image_to_sphere_func = aitoffImageToSphere
        elif proj_type.lower() == 'tan':
            self.rotator = SphericalRotator(lon_ref, lat_ref, zenithal=True)
            self.sphere_to_image_func = gnomonicSphereToImage
            self.image_to_sphere_func = gnomonicImageToSphere
        else:
            print 'WARNING: %s not recognized'%(proj_type)

    def sphereToImage(self, lon, lat):
        lon_rotated, lat_rotated = self.rotator.rotate(lon, lat)
        return self.sphere_to_image_func(lon_rotated, lat_rotated)
        
    def imageToSphere(self, x, y):
        lon_rotated, lat_rotated = self.image_to_sphere_func(x, y)
        return self.rotator.rotate(lon_rotated, lat_rotated, invert = True)

############################################################

def aitoffSphereToImage(lon, lat):
    """
    Hammer-Aitoff projection (deg).
    """
    lon = (lon < 180.) * lon + (lon > 180.) * ((lon % 360.) - 360.) # Convert angle to [-180, 180] interval
    lon = numpy.radians(lon)
    lat = numpy.radians(lat)
    gamma = (180. / numpy.pi) * numpy.sqrt(2. / (1. + (numpy.cos(lat) * numpy.cos(lon / 2.))))
    x = 2. * gamma * numpy.cos(lat) * numpy.sin(lon / 2.)
    y = gamma * numpy.sin(lat)
    return x, y

def aitoffImageToSphere(x, y):
    """
    Inverse Hammer-Aitoff projection (deg).
    """
    x = (x < 180.) * x + (x > 180.) * ((x % 360.) - 360.) # Convert angle to [-180, 180] interval
    x = numpy.array(x)
    y = numpy.array(y)
    z = numpy.sqrt(1. - ((numpy.pi / 180.) * (x / 4.))**2 - ((numpy.pi / 180.) * (y / 2.))**2) # deg
    lon = 2. * numpy.arctan2((2. * z**2) - 1,
                             (numpy.pi / 180.) * (z / 2.) * x)
    lat = numpy.arcsin((numpy.pi / 180.) * y * z)
    return ((180. - numpy.degrees(lon)) % 360.), numpy.degrees(lat)

############################################################

def gnomonicSphereToImage(lon, lat):
    """
    Gnomonic projection (deg).
    """
    lon = (lon < 180.) * lon + (lon > 180.) * ((lon % 360.) - 360.) # Convert angle to [-180, 180] interval
    lon = numpy.radians(lon)
    lat = numpy.radians(lat)
    r_theta = (180. / numpy.pi) / numpy.tan(lat)
    x = r_theta * numpy.cos(lon)
    y = r_theta * numpy.sin(lon)
    return x, y

def gnomonicImageToSphere(x, y):
    """
    Inverse gnomonic projection (deg).
    """
    x = (x < 180.) * x + (x > 180.) * ((x % 360.) - 360.) # Convert angle to [-180, 180] interval
    x = numpy.array(x)
    y = numpy.array(y)
    lon = numpy.degrees(numpy.arctan2(y, x))
    r_theta = numpy.sqrt(x**2 + y**2)
    lat = numpy.degrees(numpy.arctan(180. / (numpy.pi * r_theta)))
    return lon, lat

############################################################

#def angsep(lon_1, lat_1, lon_2, lat_2):
#    """
#    Angular separation (deg) between two sky coordinates.
#    """
#    lon_1 = numpy.radians(lon_1)
#    lat_1 = numpy.radians(lat_1)
#    lon_2 = numpy.radians(lon_2)
#    lat_2 = numpy.radians(lat_2)
#
#    mu = (numpy.cos(lat_1) * numpy.cos(lon_1) * numpy.cos(lat_2) * numpy.cos(lon_2)) \
#         + (numpy.cos(lat_1) * numpy.sin(lon_1) * numpy.cos(lat_2) * numpy.sin(lon_2)) \
#         + (numpy.sin(lat_1) * numpy.sin(lat_2))
#
#    mu = numpy.clip(mu, -1., 1.)
#
#    return numpy.degrees(numpy.arccos(mu))

def angsep(lon_1, lat_1, lon_2, lat_2):
    """
    Angular separation (deg) between two sky coordinates.
    """
    v10, v11, v12 = healpy.ang2vec(numpy.radians(90. - lat_1), numpy.radians(lon_1)).transpose()
    v20, v21, v22 = healpy.ang2vec(numpy.radians(90. - lat_2), numpy.radians(lon_2)).transpose()
    val = (v10 * v20) + (v11 * v21) + (v12 * v22)
    val = numpy.clip(val, -1., 1.)
    return numpy.degrees(numpy.arccos(val))

############################################################

def galToCel(ll, bb):
    """
    Converts Galactic (deg) to Celestial J2000 (deg) coordinates
    """
    bb = numpy.radians(bb)
    ll = numpy.radians(ll)
    ra_gp = numpy.radians(192.85948)
    de_gp = numpy.radians(27.12825)
    lcp = numpy.radians(122.932)
    sin_d = (numpy.sin(de_gp) * numpy.sin(bb)) \
            + (numpy.cos(de_gp) * numpy.cos(bb) * numpy.cos(lcp - ll))
    ramragp = numpy.arctan2(numpy.cos(bb) * numpy.sin(lcp - ll),
                            (numpy.cos(de_gp) * numpy.sin(bb)) \
                            - (numpy.sin(de_gp) * numpy.cos(bb) * numpy.cos(lcp - ll)))
    dec = numpy.arcsin(sin_d)
    ra = (ramragp + ra_gp + (2. * numpy.pi)) % (2. * numpy.pi)
    return numpy.degrees(ra), numpy.degrees(dec)

def celToGal(ra, dec):
    """
    Converts Celestial J2000 (deg) to Calactic (deg) coordinates
    """
    dec = numpy.radians(dec)
    ra = numpy.radians(ra)    
    ra_gp = numpy.radians(192.85948)
    de_gp = numpy.radians(27.12825)
    lcp = numpy.radians(122.932)    
    sin_b = (numpy.sin(de_gp) * numpy.sin(dec)) \
            + (numpy.cos(de_gp) * numpy.cos(dec) * numpy.cos(ra - ra_gp))
    lcpml = numpy.arctan2(numpy.cos(dec) * numpy.sin(ra - ra_gp),
                          (numpy.cos(de_gp) * numpy.sin(dec)) \
                          - (numpy.sin(de_gp) * numpy.cos(dec) * numpy.cos(ra-ra_gp)))
    bb = numpy.arcsin(sin_b)
    ll = (lcp - lcpml + (2. * numpy.pi)) % (2. * numpy.pi)
    return numpy.degrees(ll), numpy.degrees(bb)

############################################################

def distanceToDistanceModulus(distance):
    """
    Return distance modulus for a given distance (kpc).
    """
    return 5. * (numpy.log10(distance * 1.e3) - 1.)

def distanceModulusToDistance(distance_modulus):
    """
    Return distance (kpc) for a given distance modulus.
    """
    return 10**((0.2 * distance_modulus) - 2.)

############################################################
