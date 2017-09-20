"""
Class for converting between sphere to image coordinates using map projections.

Based on Calabretta & Greisen 2002, A&A, 357, 1077-1122
http://adsabs.harvard.edu/abs/2002A%26A...395.1077C
"""

import numpy
import numpy as np

from ugali.utils.logger import logger

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
        

        cos_psi,sin_psi = numpy.cos(psi),numpy.sin(psi)
        cos_phi,sin_phi = numpy.cos(phi),numpy.sin(phi)
        cos_theta,sin_theta = numpy.cos(theta),numpy.sin(theta)

        self.rotation_matrix = numpy.matrix([
            [cos_psi * cos_phi - cos_theta * sin_phi * sin_psi,
             cos_psi * sin_phi + cos_theta * cos_phi * sin_psi,
             sin_psi * sin_theta],
            [-sin_psi * cos_phi - cos_theta * sin_phi * cos_psi,
             -sin_psi * sin_phi + cos_theta * cos_phi * cos_psi,
             cos_psi * sin_theta],
            [sin_theta * sin_phi,
             -sin_theta * cos_phi,
             cos_theta]
        ])
        
        self.inverted_rotation_matrix = numpy.linalg.inv(self.rotation_matrix)

    def cartesian(self,lon,lat):
        lon = numpy.radians(lon)
        lat = numpy.radians(lat) 
        
        x = numpy.cos(lat) * numpy.cos(lon)
        y = numpy.cos(lat) * numpy.sin(lon)
        z =  numpy.sin(lat)
        return numpy.array([x,y,z])
        

    def rotate(self, lon, lat, invert=False):
        vec = self.cartesian(lon,lat)

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
        elif proj_type.lower() == 'car':
            def rotate(lon,lat,invert=False):
                if invert:
                    return lon + np.array([lon_ref]), lat + np.array([lat_ref])
                else:
                    return lon - np.array([lon_ref]), lat - np.array([lat_ref])
            self.rotator = SphericalRotator(lon_ref, lat_ref, zenithal=False)
            # Monkey patch the rotate function
            self.rotator.rotate = rotate 
            self.sphere_to_image_func = cartesianSphereToImage
            self.image_to_sphere_func = cartesianImageToSphere
        else:
            logger.warn('%s not recognized'%(proj_type))

    def sphereToImage(self, lon, lat):
        lon_rotated, lat_rotated = self.rotator.rotate(lon, lat)
        return self.sphere_to_image_func(lon_rotated, lat_rotated)

    sphere2image = sphereToImage
        
    def imageToSphere(self, x, y):
        lon_rotated, lat_rotated = self.image_to_sphere_func(x, y)
        return self.rotator.rotate(lon_rotated, lat_rotated, invert = True)

    image2sphere = imageToSphere

def sphere2image(lon_ref,lat_ref,lon,lat):
    proj = Projector(lon_ref,lat_ref)
    return proj.sphere2image(lon,lat)

def image2sphere(lon_ref,lat_ref,x,y):
    proj = Projector(lon_ref,lat_ref)
    return proj.image2sphere(x,y)

############################################################

# ADW: Unteseted dummy projection.
def cartesianSphereToImage(lon, lat):
    lon = lon - 360.*(lon>180)
    x,y = lon,lat
    return x,y

def cartesianImageToSphere(x,y):
    x = x - 360.*(x>180)
    lon,lat = x,y
    return lon,lat

### ADW: Speed up and fixed some issues for conversion of
### lon = 180 and lon = 360 (returned by rotator)
def aitoffSphereToImage(lon, lat):
    """
    Hammer-Aitoff projection (deg).
    """
    lon = lon - 360.*(lon>180)
    lon = numpy.radians(lon)
    lat = numpy.radians(lat)

    half_lon = lon/2.
    cos_lat = numpy.cos(lat)
     
    gamma = (180. / numpy.pi) * numpy.sqrt(2. / (1. + (cos_lat * numpy.cos(half_lon))))
    x = 2. * gamma * cos_lat * numpy.sin(half_lon)
    y = gamma * numpy.sin(lat)
    return x, y

def aitoffImageToSphere(x, y):
    """
    Inverse Hammer-Aitoff projection (deg).
    """
    x = x - 360.*(x>180)
    x = numpy.asarray(numpy.radians(x))
    y = numpy.asarray(numpy.radians(y))
    z = numpy.sqrt(1. - (x / 4.)**2 - (y / 2.)**2) # rad
    lon = 2. * numpy.arctan2((2. * z**2) - 1, (z / 2.) * x)
    lat = numpy.arcsin( y * z)
    return ((180. - numpy.degrees(lon)) % 360.), numpy.degrees(lat)

############################################################

def gnomonicSphereToImage(lon, lat):
    """
    Gnomonic projection (deg).
    """
    # Convert angle to [-180, 180] interval
    lon = lon - 360.*(lon>180)
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
    # Convert angle to [-180, 180] interval
    x = x - 360.*(x>180)
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    lon = numpy.degrees(numpy.arctan2(y, x))
    r_theta = numpy.sqrt(x**2 + y**2)
    lat = numpy.degrees(numpy.arctan(180. / (numpy.pi * r_theta)))
    return lon, lat

############################################################

def angsep2(lon_1, lat_1, lon_2, lat_2):
    """
    Angular separation (deg) between two sky coordinates.
    """
    import healpy

    v10, v11, v12 = healpy.ang2vec(numpy.radians(90. - lat_1), numpy.radians(lon_1)).transpose()
    v20, v21, v22 = healpy.ang2vec(numpy.radians(90. - lat_2), numpy.radians(lon_2)).transpose()
    val = (v10 * v20) + (v11 * v21) + (v12 * v22)
    val = numpy.clip(val, -1., 1.)
    return numpy.degrees(numpy.arccos(val))

def angsep(lon1,lat1,lon2,lat2):
    """
    Angular separation (deg) between two sky coordinates.
    Borrowed from astropy (www.astropy.org)

    Notes
    -----
    The angular separation is calculated using the Vincenty formula [1],
    which is slighly more complex and computationally expensive than
    some alternatives, but is stable at at all distances, including the
    poles and antipodes.

    [1] http://en.wikipedia.org/wiki/Great-circle_distance
    """
    lon1,lat1 = numpy.radians([lon1,lat1])
    lon2,lat2 = numpy.radians([lon2,lat2])
    
    sdlon = numpy.sin(lon2 - lon1)
    cdlon = numpy.cos(lon2 - lon1)
    slat1 = numpy.sin(lat1)
    slat2 = numpy.sin(lat2)
    clat1 = numpy.cos(lat1)
    clat2 = numpy.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return numpy.degrees(numpy.arctan2(numpy.hypot(num1,num2), denominator))

############################################################

# ADW: Reduce numpy array operations for speed
def galToCel(ll, bb):
    """
    Converts Galactic (deg) to Celestial J2000 (deg) coordinates
    """
    bb = numpy.radians(bb)
    sin_bb = numpy.sin(bb)
    cos_bb = numpy.cos(bb)

    ll = numpy.radians(ll)
    ra_gp = numpy.radians(192.85948)
    de_gp = numpy.radians(27.12825)
    lcp = numpy.radians(122.932)

    sin_lcp_ll = numpy.sin(lcp - ll)
    cos_lcp_ll = numpy.cos(lcp - ll)

    sin_d = (numpy.sin(de_gp) * sin_bb) \
            + (numpy.cos(de_gp) * cos_bb * cos_lcp_ll)
    ramragp = numpy.arctan2(cos_bb * sin_lcp_ll,
                            (numpy.cos(de_gp) * sin_bb) \
                            - (numpy.sin(de_gp) * cos_bb * cos_lcp_ll))
    dec = numpy.arcsin(sin_d)
    ra = (ramragp + ra_gp + (2. * numpy.pi)) % (2. * numpy.pi)
    return numpy.degrees(ra), numpy.degrees(dec)

gal2cel = galToCel

def celToGal(ra, dec):
    """
    Converts Celestial J2000 (deg) to Calactic (deg) coordinates
    """
    dec = numpy.radians(dec)
    sin_dec = numpy.sin(dec)
    cos_dec = numpy.cos(dec)

    ra = numpy.radians(ra)    
    ra_gp = numpy.radians(192.85948)
    de_gp = numpy.radians(27.12825)

    sin_ra_gp = numpy.sin(ra - ra_gp)
    cos_ra_gp = numpy.cos(ra - ra_gp)

    lcp = numpy.radians(122.932)    
    sin_b = (numpy.sin(de_gp) * sin_dec) \
            + (numpy.cos(de_gp) * cos_dec * cos_ra_gp)
    lcpml = numpy.arctan2(cos_dec * sin_ra_gp,
                          (numpy.cos(de_gp) * sin_dec) \
                          - (numpy.sin(de_gp) * cos_dec * cos_ra_gp))
    bb = numpy.arcsin(sin_b)
    ll = (lcp - lcpml + (2. * numpy.pi)) % (2. * numpy.pi)
    return numpy.degrees(ll), numpy.degrees(bb)

cel2gal = celToGal

def estimate_angle(angle, origin, new_frame, offset=1e-7):
    """
    https://github.com/astropy/astropy/issues/3093
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    angle_deg = angle*np.pi/180
    newlat = offset * np.cos(angle_deg) + origin.data.lat.degree
    newlon = (offset * np.sin(angle_deg) / np.cos(newlat * np.pi/180) + origin.data.lon.degree)
    sc = SkyCoord(newlon, newlat, unit='degree', frame=origin.frame.name)
    new_origin = origin.transform_to(new_frame)
    new_sc = sc.transform_to(new_frame)
    return new_origin.position_angle(new_sc).deg

def gal2cel_angle(glon,glat,angle,offset=1e-7):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    origin = SkyCoord(glon,glat,unit=u.deg,frame='galactic')
    return estimate_angle(angle,origin,'fk5',offset)

def cel2gal_angle(ra,dec,angle,offset=1e-7):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    origin = SkyCoord(ra,dec,unit=u.deg,frame='fk5')
    return estimate_angle(angle,origin,'galactic',offset)

### ADW: Probably works, remember 90-pa for kernel convention...
### def gal2cel_angle(glon,glat,angle):
###     """
###     WARNING: Not vectorized
###     """
###     gal_angle = np.radians(angle)
###     galproj = Projector(glon,glat)
###     x,y = np.sin(gal_angle),np.cos(gal_angle)
###     alon,alat = galproj.imageToSphere(0.1*x,0.1*y)
###  
###     ra,dec = gal2cel(glon,glat)
###     ara,adec = gal2cel(alon,alat)
###     celproj = Projector(ra,dec)
###     cel_x,cel_y = celproj.sphereToImage(ara,adec)
###     cel_angle = np.degrees(np.arctan2(cel_x,cel_y))
###     return cel_angle + 360*(cel_angle<0)


### ADW: WARNING DOESN'T WORK YET
### def cel2gal_angle(ra,dec,angle):
###     """
###     WARNING: Not vectorized
###     """
###     cel_angle = np.radians(angle)
###     celproj = Projector(ra,dec)
###     x,y = np.sin(cel_angle),np.cos(cel_angle)
###     angle_ra,angle_dec = celproj.imageToSphere(1e-2*x,1e-2*y)
###  
###     glon,glat = gal2cel(ra,dec)
###     angle_glon,angle_glat = cel2gal(angle_ra,angle_dec)
###     galproj = Projector(glon,glat)
###     gal_x,gal_y = galproj.sphereToImage(angle_glon,angle_glat)
###     gal_angle = np.degrees(np.arctan2(gal_x,gal_y))
###     return gal_angle + 360*(gal_angle<0)


############################################################

def dec2hms(dec):
    """
    ADW: This should really be replaced by astropy
    """
    DEGREE = 360.
    HOUR = 24.
    MINUTE = 60.
    SECOND = 3600.
    
    if isinstance(dec,basestring):
        dec = float(dec)

    fhour = dec*(HOUR/DEGREE)
    hour = int(fhour)

    fminute = (fhour - hour)*MINUTE
    minute = int(fminute)
    
    second = (fminute - minute)*MINUTE
    return (hour, minute, second)

def dec2dms(dec):
    """
    ADW: This should really be replaced by astropy
    """
    DEGREE = 360.
    HOUR = 24.
    MINUTE = 60.
    SECOND = 3600.
    
    if isinstance(dec,basestring):
        dec = float(dec)

    sign = numpy.copysign(1.0,dec)

    fdeg = np.abs(dec)
    deg = int(fdeg)
    
    fminute = (fdeg - deg)*MINUTE
    minute = int(fminute)
    
    second = (fminute - minute)*MINUTE

    deg = int(deg * sign)
    return (deg, minute, second)

def hms2dec(hms):
    """
    Convert longitude from hours,minutes,seconds in string or 3-array
    format to decimal degrees.

    ADW: This really should be replaced by astropy
    """
    DEGREE = 360.
    HOUR = 24.
    MINUTE = 60.
    SECOND = 3600.

    if isinstance(hms,basestring):
        hour,minute,second = numpy.array(re.split('[hms]',hms))[:3].astype(float)
    else:
        hour,minute,second = hms.T

    decimal = (hour + minute * 1./MINUTE + second * 1./SECOND)*(DEGREE/HOUR)
    return decimal

def dms2dec(dms):
    """
    Convert latitude from degrees,minutes,seconds in string or 3-array
   format to decimal degrees.
    """
    DEGREE = 360.
    HOUR = 24.
    MINUTE = 60.
    SECOND = 3600.

    # Be careful here, degree needs to be a float so that negative zero
    # can have its signbit set:
    # http://docs.scipy.org/doc/numpy-1.7.0/reference/c-api.coremath.html#NPY_NZERO

    if isinstance(dms,basestring):
        degree,minute,second = numpy.array(re.split('[dms]',hms))[:3].astype(float)
    else:
        degree,minute,second = dms.T

    sign = numpy.copysign(1.0,degree)
    decimal = numpy.abs(degree) + minute * 1./MINUTE + second * 1./SECOND
    decimal *= sign
    return decimal

def sr2deg(solid_angle):
    return np.degrees(np.degrees(solid_angle))

def deg2sr(solid_angle):
    return np.radians(np.radians(solid_angle))

############################################################

def distanceToDistanceModulus(distance):
    """
    Return distance modulus for a given distance (kpc).
    """
    return 5. * (numpy.log10(numpy.array(distance) * 1.e3) - 1.)

dist2mod = distanceToDistanceModulus

def distanceModulusToDistance(distance_modulus):
    """
    Return distance (kpc) for a given distance modulus.
    """
    return 10**((0.2 * numpy.array(distance_modulus)) - 2.)

mod2dist = distanceModulusToDistance

############################################################

def ang2const(lon,lat,coord='gal'):
    import ephem

    scalar = np.isscalar(lon)
    lon = np.array(lon,copy=False,ndmin=1)
    lat = np.array(lat,copy=False,ndmin=1)

    if coord.lower() == 'cel':
        ra,dec = lon,lat
    elif coord.lower() == 'gal':
        ra,dec = gal2cel(lon,lat)
    else:
        msg = "Unrecognized coordinate"
        raise Exception(msg)

    x,y = np.radians([ra,dec])
    const = [ephem.constellation(coord) for coord in zip(x,y)]
    if scalar: return const[0]
    return const

def ang2iau(lon,lat,coord='gal'):
    # Default name formatting
    # http://cdsarc.u-strasbg.fr/ftp/pub/iau/
    # http://cds.u-strasbg.fr/vizier/Dic/iau-spec.htx
    fmt = "J%(hour)02i%(hmin)02i%(deg)+03i%(dmin)02i"

    scalar = np.isscalar(lon)
    lon = np.array(lon,copy=False,ndmin=1)
    lat = np.array(lat,copy=False,ndmin=1)

    if coord.lower() == 'cel':
        ra,dec = lon,lat
    elif coord.lower() == 'gal':
        ra,dec = gal2cel(lon,lat)
    else:
        msg = "Unrecognized coordinate"
        raise Exception(msg)

    x,y = np.radians([ra,dec])
    iau = []
    for _ra,_dec in zip(ra,dec):
        hms = dec2hms(_ra); dms = dec2dms(_dec)
        params = dict(hour=hms[0],hmin=hms[1],
                      deg=dms[0],dmin=dms[1])
        iau.append(fmt%params)
    if scalar: return iau[0]
    return iau

    
def match(lon1, lat1, lon2, lat2, tol=None, nnearest=1):
    """
    Adapted from Eric Tollerud.
    Finds matches in one catalog to another.
 
    Parameters
    lon1 : array-like
        Longitude of the first catalog (degrees)
    lat1 : array-like
        Latitude of the first catalog (shape of array must match `lon1`)
    lon2 : array-like
        Longitude of the second catalog
    lat2 : array-like
        Latitude of the second catalog (shape of array must match `lon2`)
    tol : float or None, optional
        Proximity (degrees) of a match to count as a match.  If None,
        all nearest neighbors for the first catalog will be returned.
    nnearest : int, optional
        The nth neighbor to find.  E.g., 1 for the nearest nearby, 2 for the
        second nearest neighbor, etc.  Particularly useful if you want to get
        the nearest *non-self* neighbor of a catalog.  To do this, use:
        ``spherematch(lon, lat, lon, lat, nnearest=2)``
 
    Returns
    -------
    idx1 : int array
        Indices into the first catalog of the matches. Will never be
        larger than `lon1`/`lat1`.
    idx2 : int array
        Indices into the second catalog of the matches. Will never be
        larger than `lon2`/`lat2`.
    ds : float array
        Distance (in degrees) between the matches
    """
    from scipy.spatial import cKDTree
 
    lon1 = numpy.asarray(lon1)
    lat1 = numpy.asarray(lat1)
    lon2 = numpy.asarray(lon2)
    lat2 = numpy.asarray(lat2)
 
    if lon1.shape != lat1.shape:
        raise ValueError('lon1 and lat1 do not match!')
    if lon2.shape != lat2.shape:
        raise ValueError('lon2 and lat2 do not match!')

    rotator = SphericalRotator(0,0)

 
    # This is equivalent, but faster than just doing numpy.array([x1, y1, z1]).T
    x1, y1, z1 = rotator.cartesian(lon1.ravel(),lat1.ravel())
    coords1 = numpy.empty((x1.size, 3))
    coords1[:, 0] = x1
    coords1[:, 1] = y1
    coords1[:, 2] = z1
 
    x2, y2, z2 = rotator.cartesian(lon2.ravel(),lat2.ravel())
    coords2 = numpy.empty((x2.size, 3))
    coords2[:, 0] = x2
    coords2[:, 1] = y2
    coords2[:, 2] = z2
 
    tree = cKDTree(coords2)
    if nnearest == 1:
        idxs2 = tree.query(coords1)[1]
    elif nnearest > 1:
        idxs2 = tree.query(coords1, nnearest)[1][:, -1]
    else:
        raise ValueError('invalid nnearest ' + str(nnearest))
 
    ds = angsep(lon1, lat1, lon2[idxs2], lat2[idxs2])
 
    idxs1 = numpy.arange(lon1.size)
 
    if tol is not None:
        msk = ds < tol
        idxs1 = idxs1[msk]
        idxs2 = idxs2[msk]
        ds = ds[msk]
 
    return idxs1, idxs2, ds
