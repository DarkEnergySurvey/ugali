from numpy import array, sqrt, zeros, sin, cos, arange, arcsin,\
      arctan2, transpose, concatenate, ndarray, pi, dot, deg2rad,\
      rad2deg

def jprecess(ra, dec, mu_radec=None, parallax=None, rad_vel=None, epoch=None):
   """
    NAME:
         JPRECESS
    PURPOSE:
         Precess astronomical coordinates from B1950 to J2000
    EXPLANATION:
         Calculate the mean place of a star at J2000.0 on the FK5 system from the
         mean place at B1950.0 on the FK4 system.
    
         Use BPRECESS for the reverse direction J2000 ==> B1950
    CALLING SEQUENCE:
         jprecess, ra, dec, ra_2000, dec_2000, [ MU_RADEC = , PARALLAX = 
                  RAD_VEL =, EPOCH =   ]
    
    INPUTS:
         RA,DEC - input B1950 right ascension and declination in *degrees*.
                  Scalar or vector
    
    OUTPUTS:
         RA_2000, DEC_2000 - the corresponding J2000 right ascension and 
                  declination in *degrees*.   Same number of elements as RA,DEC
                  but always double precision. 
    
    OPTIONAL INPUT-OUTPUT KEYWORDS
         MU_RADEC - 2xN element double precision vector containing the proper 
                     motion in seconds of arc per tropical *century* in right 
                     ascension and declination.
         PARALLAX - N_element vector giving stellar parallax (seconds of arc)
         RAD_VEL  - N_element vector giving radial velocity in km/s
    
          The values of MU_RADEC, PARALLAX, and RADVEL will all be modified
          upon output to contain the values of these quantities in the
          J2000 system.    Values will also be converted to double precision.  
          The parallax and radial velocity will have a very minor influence on 
          the J2000 position.
    
          EPOCH - scalar giving epoch of original observations, default 1950.0d
              This keyword value is only used if the MU_RADEC keyword is not set.
     NOTES:
          The algorithm is taken from the Explanatory Supplement to the 
          Astronomical Almanac 1992, page 184.
          Also see Aoki et al (1983), A&A, 128,263
    
          JPRECESS distinguishes between the following two cases:
          (1) The proper motion is known and non-zero
          (2) the proper motion is unknown or known to be exactly zero (i.e.
                  extragalactic radio sources).   In this case, the algorithm
                  in Appendix 2 of Aoki et al. (1983) is used to ensure that
                  the output proper motion is  exactly zero.    Better precision
                  can be achieved in this case by inputting the EPOCH of the
                  original observations.
    
          The error in using the IDL procedure PRECESS for converting between
          B1950 and J2000 can be up to 12", mainly in right ascension.   If
          better accuracy than this is needed then JPRECESS should be used.
    
    EXAMPLE:
          The SAO catalogue gives the B1950 position and proper motion for the 
          star HD 119288.   Find the J2000 position. 
    
             RA(1950) = 13h 39m 44.526s      Dec(1950) = 8d 38' 28.63''  
             Mu(RA) = -.0259 s/yr      Mu(Dec) = -.093 ''/yr
    
          IDL> mu_radec = 100D* [ -15D*.0259, -0.093 ]
          IDL> ra = ten(13,39,44.526)*15.D 
          IDL> dec = ten(8,38,28.63)
          IDL> jprecess, ra, dec, ra2000, dec2000, mu_radec = mu_radec
          IDL> print, adstring(ra2000, dec2000,2)
                  ===> 13h 42m 12.740s    +08d 23' 17.69"
    
    RESTRICTIONS:
         "When transferring individual observations, as opposed to catalog mean
          place, the safest method is to tranform the observations back to the
          epoch of the observation, on the FK4 system (or in the system that was
          used to to produce the observed mean place), convert to the FK5 system,
          and transform to the the epoch and equinox of J2000.0" -- from the
          Explanatory Supplement (1992), p. 180
    
    REVISION HISTORY:
          Written,    W. Landsman                September, 1992
          Corrected a couple of typos in M matrix   October, 1992
          Vectorized, W. Landsman                   February, 1994
          Implement Appendix 2 of Aoki et al. (1983) for case where proper
          motion unknown or exactly zero     W. Landsman    November, 1994
          Converted to IDL V5.0   W. Landsman   September 1997
          Fixed typo in updating proper motion   W. Landsman   April 1999
          Make sure proper motion is floating point  W. Landsman December 2000
          Use V6.0 notation  W. Landsman Mar 2011
          Converted to python by A. Drlica-Wagner Feb 2014
   """
   if isinstance(ra, ndarray):
      ra = array(ra)
      dec = array(dec)
   else:
      ra = array([ra0])
      dec = array([dec0])
   n = ra.size
    
   if rad_vel is None:   
      rad_vel = zeros(n,dtype=float)
   else:
      if not isinstance(rad_vel, ndarray):
         rad_vel = array([rad_vel],dtype=float)
      if rad_vel.size != n:   
         raise Exception('ERROR - RAD_VEL keyword vector must be of the same length as RA and DEC')
    
   if (mu_radec is not None):   
      if (array(mu_radec).size != 2 * n):   
         raise Exception('ERROR - MU_RADEC keyword (proper motion) be dimensioned (2,' + strtrim(n, 2) + ')')
      mu_radec = mu_radec * 1.
    
   if parallax is None:   
      parallax = zeros(n,dtype=float)
   else:   
      if not isinstance(parallax, ndarray):
         parallax = array([parallax],dtype=float)
    
   if epoch is None:   
      epoch = 1950.0e0
    
   radeg = 180.e0 / pi
   sec_to_radian = 1/radeg/3600.
   #sec_to_radian = lambda x : deg2rad(x/3600.)
    
   m =  array([ 
      array([+0.9999256782e0, +0.0111820610e0, +0.0048579479e0,  \
                -0.000551e0,     +0.238514e0,     -0.435623e0      ]), 
      array([ -0.0111820611e0, +0.9999374784e0, -0.0000271474e0,  \
                 -0.238565e0,     -0.002667e0,      +0.012254e0    ]), 
      array([ -0.0048579477e0, -0.0000271765e0, +0.9999881997e0 , \
                 +0.435739e0,      -0.008541e0,      +0.002117e0   ]), 
      array([ +0.00000242395018e0, +0.00000002710663e0, +0.00000001177656e0, \
                 +0.99994704e0,    +0.01118251e0,    +0.00485767e0 ]), 
      array([ -0.00000002710663e0, +0.00000242397878e0, -0.00000000006582e0, \
                 -0.01118251e0,     +0.99995883e0,    -0.00002714e0]), 
      array([ -0.00000001177656e0, -0.00000000006587e0, 0.00000242410173e0, \
                 -0.00485767e0,   -0.00002718e0,     1.00000956e0  ]) ])
    
   a = 1e-6 * array([ -1.62557e0, -0.31919e0, -0.13843e0])        #in radians
   a_dot = 1e-3 * array([1.244e0, -1.579e0, -0.660e0 ])           #in arc seconds per century
    
   ra_rad = deg2rad(ra)
   dec_rad = deg2rad(dec)
   cosra = cos(ra_rad)
   sinra = sin(ra_rad)
   cosdec = cos(dec_rad)
   sindec = sin(dec_rad)
    
   ra_2000 = ra*0.
   dec_2000 = dec*0.
    
   for i in range(n):
      r0 = array([ cosra[i]*cosdec[i], sinra[i]*cosdec[i], sindec[i] ])
    
      if (mu_radec is None):   
         mu_a = 0.
         mu_d = 0.
      else:
         mu_a = mu_radec[ i, 0]
         mu_d = mu_radec[ i, 1]
    
      #Velocity vector
      r0_dot = array([-mu_a*sinra[i]*cosdec[i] - mu_d*cosra[i]*sindec[i], \
                         mu_a*cosra[i]*cosdec[i] - mu_d*sinra[i]*sindec[i] , \
                         mu_d*cosdec[i] ])  + 21.095e0 * rad_vel[i] * parallax[i] * r0
    
      r1 = r0 - a + ((a * r0).sum())*r0
      r1_dot = r0_dot - a_dot + (( a * r0).sum())*r0
    
      r_1 = concatenate((r1, r1_dot))
      r = transpose(dot(transpose(m),transpose(r_1)))
    
      if mu_radec is None:   
         rr = r[0:3] 
         v =  r[3:6] 
         t = ((epoch - 1950.0e0) - 50.00021e0)/100.0
         rr1 = rr + sec_to_radian*v*t
         x = rr1[0]  ; y = rr1[1]  ; z = rr1[2]  
      else:
         x = r[0]  ; y = r[1]  ; z = r[2]  
         x_dot = r[3]  ; y_dot= r[4]  ;  z_dot = r[5]
    
      r2 = x**2 + y**2 + z**2
      rmag = sqrt( r2 )
      dec_2000[i] = arcsin(z / rmag)
      ra_2000[i] = arctan2(y, x)
    
      if mu_radec is not None:
         mu_radec[i, 0] = ( x*y_dot - y*x_dot) / ( x**2 + y**2)
         mu_radec[i, 1] = ( z_dot* (x**2 + y**2) - z*(x*x_dot + y*y_dot) ) /  \
                     ( r2*sqrt( x**2 + y**2) )
    
      if parallax[i] > 0.:
         rad_vel[i] = ( x*x_dot + y*y_dot + z*z_dot )/ (21.095*parallax[i]*rmag)
         parallax[i] = parallax[i] / rmag
    
   neg = (ra_2000 < 0)
   if neg.any() > 0:
      ra_2000[neg] = ra_2000[neg] + 2.0 * pi
      
   ra_2000 = ra_2000*radeg ; dec_2000 = dec_2000*radeg
    
   if ra.size == 1:
      ra_2000 = ra_2000[0]     ; dec_2000 = dec_2000[0]
    
   return ra_2000, dec_2000


def bprecess(ra0, dec0, mu_radec=None, parallax=None, rad_vel=None, epoch=None):
   """
    NAME:
          BPRECESS
    PURPOSE:
          Precess positions from J2000.0 (FK5) to B1950.0 (FK4)
    EXPLANATION:
          Calculates the mean place of a star at B1950.0 on the FK4 system from
          the mean place at J2000.0 on the FK5 system.
   
    CALLING SEQUENCE:
          bprecess, ra, dec, ra_1950, dec_1950, [ MU_RADEC = , PARALLAX =
                                          RAD_VEL =, EPOCH =   ]
   
    INPUTS:
          RA,DEC - Input J2000 right ascension and declination in *degrees*.
                  Scalar or N element vector
   
    OUTPUTS:
          RA_1950, DEC_1950 - The corresponding B1950 right ascension and
                  declination in *degrees*.    Same number of elements as
                  RA,DEC but always double precision.
   
    OPTIONAL INPUT-OUTPUT KEYWORDS
          MU_RADEC - 2xN element double precision vector containing the proper
                     motion in seconds of arc per tropical *century* in right
                     ascension and declination.
          PARALLAX - N_element vector giving stellar parallax (seconds of arc)
          RAD_VEL  - N_element vector giving radial velocity in km/s
   
          The values of MU_RADEC, PARALLAX, and RADVEL will all be modified
          upon output to contain the values of these quantities in the
          B1950 system.  The parallax and radial velocity will have a very
          minor influence on the B1950 position.
   
          EPOCH - scalar giving epoch of original observations, default 2000.0d
              This keyword value is only used if the MU_RADEC keyword is not set.
    NOTES:
          The algorithm is taken from the Explanatory Supplement to the
          Astronomical Almanac 1992, page 186.
          Also see Aoki et al (1983), A&A, 128,263
   
          BPRECESS distinguishes between the following two cases:
          (1) The proper motion is known and non-zero
          (2) the proper motion is unknown or known to be exactly zero (i.e.
                  extragalactic radio sources).   In this case, the reverse of
                  the algorithm in Appendix 2 of Aoki et al. (1983) is used to
                  ensure that the output proper motion is  exactly zero. Better
                  precision can be achieved in this case by inputting the EPOCH
                  of the original observations.
   
          The error in using the IDL procedure PRECESS for converting between
          B1950 and J1950 can be up to 12", mainly in right ascension.   If
          better accuracy than this is needed then BPRECESS should be used.
   
          An unsystematic comparison of BPRECESS with the IPAC precession
          routine (http://nedwww.ipac.caltech.edu/forms/calculator.html) always
          gives differences less than 0.15".
    EXAMPLE:
          The SAO2000 catalogue gives the J2000 position and proper motion for
          the star HD 119288.   Find the B1950 position.
   
          RA(2000) = 13h 42m 12.740s      Dec(2000) = 8d 23' 17.69''
          Mu(RA) = -.0257 s/yr      Mu(Dec) = -.090 ''/yr
   
          IDL> mu_radec = 100D* [ -15D*.0257, -0.090 ]
          IDL> ra = ten(13, 42, 12.740)*15.D
          IDL> dec = ten(8, 23, 17.69)
          IDL> bprecess, ra, dec, ra1950, dec1950, mu_radec = mu_radec
          IDL> print, adstring(ra1950, dec1950,2)
                  ===> 13h 39m 44.526s    +08d 38' 28.63"
   
    REVISION HISTORY:
          Written,    W. Landsman                October, 1992
          Vectorized, W. Landsman                February, 1994
          Treat case where proper motion not known or exactly zero  November 1994
          Handling of arrays larger than 32767   Lars L. Christensen, march, 1995
          Converted to IDL V5.0   W. Landsman   September 1997
          Fixed bug where A term not initialized for vector input
               W. Landsman        February 2000
         Converted to python 			Sergey Koposov july 2010   
   """

   scal = True
   if isinstance(ra0, ndarray):
      ra = ra0
      dec = dec0
      n = ra.size
      scal = False
   else:
      n = 1
      ra = array([ra0])
      dec = array([dec0])
      
   if rad_vel is None:   
      rad_vel = zeros(n)
   else:
      if not isinstance(rad_vel, ndarray):
         rad_vel = array([rad_vel],dtype=float)
      if rad_vel.size != n:   
         raise Exception('ERROR - RAD_VEL keyword vector must be of the same length as RA and DEC')
   
   if (mu_radec is not None):   
      if (array(mu_radec).size != 2 * n):   
         raise Exception('ERROR - MU_RADEC keyword (proper motion) be dimensioned (2,' + strtrim(n, 2) + ')')
      mu_radec = mu_radec * 1.
   
   if parallax is None:   
      parallax = zeros(n)
   else:   
      if not isinstance(parallax, ndarray):
         parallax = array([parallax],dtype=float)
   
   if epoch is None:   
      epoch = 2000.0e0
   
   radeg = 180.e0 / pi
   sec_to_radian = lambda x : deg2rad(x/3600.)
   
   m = array([array([+0.9999256795e0, -0.0111814828e0, -0.0048590040e0, -0.000551e0, -0.238560e0, +0.435730e0]),
   array([+0.0111814828e0, +0.9999374849e0, -0.0000271557e0, +0.238509e0, -0.002667e0, -0.008541e0]),
   array([+0.0048590039e0, -0.0000271771e0, +0.9999881946e0, -0.435614e0, +0.012254e0, +0.002117e0]),
   array([-0.00000242389840e0, +0.00000002710544e0, +0.00000001177742e0, +0.99990432e0, -0.01118145e0, -0.00485852e0]),
   array([-0.00000002710544e0, -0.00000242392702e0, +0.00000000006585e0, +0.01118145e0, +0.99991613e0, -0.00002716e0]),
   array([-0.00000001177742e0, +0.00000000006585e0, -0.00000242404995e0, +0.00485852e0, -0.00002717e0, +0.99996684e0])])
   
   a_dot = 1e-3 * array([1.244e0, -1.579e0, -0.660e0])           #in arc seconds per century
   
   ra_rad = deg2rad(ra)
   dec_rad = deg2rad(dec)
   cosra = cos(ra_rad)
   sinra = sin(ra_rad)
   cosdec = cos(dec_rad)
   sindec = sin(dec_rad)
   
   dec_1950 = dec * 0.
   ra_1950 = ra * 0.
   
   for i in range(n):
   
   # Following statement moved inside loop in Feb 2000.
      a = 1e-6 * array([-1.62557e0, -0.31919e0, -0.13843e0])        #in radians
      
      r0 = array([cosra[i] * cosdec[i], sinra[i] * cosdec[i], sindec[i]])
      
      if (mu_radec is not None):   
         
         mu_a = mu_radec[i,0]
         mu_d = mu_radec[i,1]
         r0_dot = array([-mu_a * sinra[i] * cosdec[i] - mu_d * cosra[i] * sindec[i], mu_a * cosra[i] * cosdec[i] - mu_d * sinra[i] * sindec[i], mu_d * cosdec[i]]) + 21.095e0 * rad_vel[i] * parallax[i] * r0
         
      else:   
         r0_dot = array([0.0e0, 0.0e0, 0.0e0])
      
      r_0 = concatenate((r0, r0_dot))
      r_1 = transpose(dot(transpose(m), transpose(r_0)))
      
      # Include the effects of the E-terms of aberration to form r and r_dot.
      
      r1 = r_1[0:3]
      r1_dot = r_1[3:6]
      
      if mu_radec is None:   
         r1 = r1 + sec_to_radian ( r1_dot * (epoch - 1950.0e0) / 100. )
         a = a + sec_to_radian ( a_dot * (epoch - 1950.0e0) / 100. )
      
      x1 = r_1[0]   ;   y1 = r_1[1]    ;  z1 = r_1[2]
      rmag = sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)
      
      
      s1 = r1 / rmag    ; s1_dot = r1_dot / rmag
      
      s = s1
      for j in arange(0, 3):
         r = s1 + a - ((s * a).sum()) * s
         s = r / rmag
      x = r[0]          ; y = r[1]     ;  z = r[2]
      r2 = x ** 2 + y ** 2 + z ** 2
      rmag = sqrt(r2)
      
      if mu_radec is not None:   
         r_dot = s1_dot + a_dot - ((s * a_dot).sum()) * s
         x_dot = r_dot[0]  ; y_dot = r_dot[1]  ;  z_dot = r_dot[2]
         mu_radec[i,0] = (x * y_dot - y * x_dot) / (x ** 2 + y ** 2)
         mu_radec[i,1] = (z_dot * (x ** 2 + y ** 2) - z * (x * x_dot + y * y_dot)) / (r2 * sqrt(x ** 2 + y ** 2))
      
      dec_1950[i] = arcsin(z / rmag)
      ra_1950[i] = arctan2(y, x)
      
      if parallax[i] > 0.:   
         rad_vel[i] = (x * x_dot + y * y_dot + z * z_dot) / (21.095 * parallax[i] * rmag)
         parallax[i] = parallax[i] / rmag
   
   neg = (ra_1950 < 0)
   if neg.any() > 0:   
      ra_1950[neg] = ra_1950[neg] + 2.e0 * pi
   
   ra_1950 = rad2deg(ra_1950)
   dec_1950 = rad2deg(dec_1950)
   
   # Make output scalar if input was scalar
   if scal:
      return ra_1950[0],dec_1950[0]
   else:
      return ra_1950, dec_1950

if __name__ == '__main__':
    from numpy import allclose

    ra  = array([  0., 9., 19., 29., 39., 49., 59., 69., 79., 89.])
    dec = array([ 0., 9., 19., 29., 39., 49., 59., 69., 79., 89.])
    ra2000  = array([0.64069095, 9.6479274, 19.672634, 29.716674, 39.784237, 49.884683, 60.040969, 70.322221, 81.054046, 105.23936])
    dec2000 = array([0.27840945, 9.2747207, 19.262679, 29.242615, 39.215111, 49.180972, 59.141167, 69.096721, 79.048183, 88.965161])
    ra1950  = array([ 359.35928, 8.3527459, 18.328867, 28.285743, 38.219246, 48.120124, 57.965657, 67.687386, 76.962913, 72.909428])
    dec1950 = array([-0.27834947, 8.7248461, 18.736315, 28.755752, 38.782542, 48.815833, 58.854545, 68.89732, 78.94207, 88.955871])

    print "Testing jprecess..."
    test_ra2000, test_dec2000 = jprecess(ra,dec)
    print " RAs:",  allclose(ra2000,test_ra2000)
    print " DECs:", allclose(dec2000,test_dec2000)
    print "Testing bprecess..."
    test_ra1950, test_dec1950 = bprecess(ra,dec)
    print " RAs:",  allclose(ra1950,test_ra1950)
    print " DECs:", allclose(dec1950,test_dec1950)
