{
 'output': {'chatter': 4, # 0 = silent; 1 = text output; 2 = more text output; 3 = text output and figures; 4 = paranoid
            'savedir_mag_1_mask': './mask_g',
            'savedir_mag_2_mask': './mask_r',
            'savedir_likelihood': './likelihood',
            'logdir_likelihood': './likelihood/log',
 },

 'queue': {'cluster': 'slac',
           'script': '/u/ki/kadrlica/des/software/ugali/analysis/scan.py',
           'jobname': 'ugali',
           'queue': 'long',
           'require': 'scratch > 1 && rhel60',
           'max_jobs': 250,
 },

 'catalog': {
     'dirname': '/u/ki/kadrlica/sdss/data/dr10/healpix/',
     'basename': """catalog_hpx%04i.fits""",
     'lon_field': 'GLON',
     'lat_field': 'GLAT',
     'coordsys': 'gal',
     'mag_1_field': 'MAG_PSF_SFD_G', # Color is defined as mag_1 - mag_2
     'mag_err_1_field': 'MAGERR_PSF_G',
     'mag_2_field': 'MAG_PSF_SFD_R',
     'mag_err_2_field': 'MAGERR_PSF_R',
     'mc_source_id_field': 'MC_SOURCE_ID',
     #'mc_source_id_field': None,
     'band_1_detection': True # True = band 1 is detection band; False = band 2 is detection band
 }, 

 'mangle': {'infile_1': '/u/ki/kadrlica/sdss/data/dr10/mangle/sdss.dr10.holy.uni.ply',
            'infile_2': '/u/ki/kadrlica/sdss/data/dr10/mangle/sdss.dr10.holy.uni.ply',
            'coordsys': 'gal'},
 #'manglebindir': '/home/s1/bechtol/des10.a/code/mangle/mangle2.2/bin'},

 'mask': {
     'dirname'    : '/u/ki/kadrlica/sdss/data/dr10/simple',
     'basename_1' : """maglim_g_hpx%04i.fits""",
     'basename_2' : """maglim_r_hpx%04i.fits""",
     'infile_1': '/u/gl/bechtol/disk/DES/mw_substructure/sv_test/y1c2/data/mask/hpx_4096_y1c2_coadd_holymolys_maglims_g_sparse_scaled.fits',
     'infile_2': '/u/gl/bechtol/disk/DES/mw_substructure/sv_test/y1c2/data/mask/hpx_4096_y1c2_coadd_holymolys_maglims_r_sparse_scaled.fits',
     'minimum_solid_angle': 0.1 # deg^2
 }, 
          
 'color': {
     'min': -0.5,
     'max': 1.0, # 1.0
     'n_bins': 12 # 15
 }, 

 'mag': {
     'min': 18.5, #18
     'max': 23, # 25
     'n_bins': 70 # 0.1, 0.02
 }, 

 # HEALPix nside choices
 # 2^0  -> 3518 arcmin,   3438 deg^2, 12e6 arcmin^2
 # 2^1  -> 1759 arcmin,   859. deg^2, 31e5 arcmin^2
 # 2^2  -> 879. arcmin,   214. deg^2, 77e4 arcmin^2
 # 2^3  -> 440. arcmin,   53.7 deg^2, 19e4 arcmin^2
 # 2^4  -> 220. arcmin,   13.4 deg^2, 48e3 arcmin^2
 # 2^5  -> 110. arcmin,   3.36 deg^2, 12e3 arcmin^2
 # 2^6  -> 55.0 arcmin,   0.84 deg^2, 3021 arcmin^2
 # 2^7  -> 27.5 arcmin,   0.21 deg^2,  755 arcmin^2
 # 2^8  -> 13.7 arcmin,  0.052 deg^2,  189 arcmin^2
 # 2^9  -> 6.87 arcmin,  0.013 deg^2, 47.2 arcmin^2
 # 2^10 -> 3.44 arcmin, 3.3e-3 deg^2, 11.8 arcmin^2
 # 2^11 -> 1.72 arcmin, 8.2e-4 deg^2, 2.95 arcmin^2
 # 2^12 -> 0.86 arcmin, 2.0e-4 deg^2, 0.74 arcmin^2
 # 2^13 -> 0.43 arcmin, 5.1e-5 deg^2, 0.18 arcmin^2

 'coords': {
            'nside_catalog': 2**2, # Size of patches for catalog binning
            'nside_mask': 2**6, # Size of patches for mask creation
            'nside_likelihood': 2**8, # Size of patches for likelihood analysis
            'nside_pixel': 2**12, 
            'roi_radius': 1.25, # Outer radius of annulus used to make empirical background model
            'roi_radius_annulus': 0.5, # Inner radius of annulus used to make empirical background model
            'coordsys': 'gal',
            'proj_type': 'ait'
 },

 'kernel': {'type': 'plummer',
            'params': [0.10]},
 #'kernel': {'type': 'disk',
 #           'params': [0.10]},

 'data': {'survey': 'sdss',
          'release': 'dr10',
          'dirname': '/u/ki/kadrlica/sdss/data/dr10/raw',
          'density': """/u/ki/kadrlica/sdss/data/dr10/density/density_hpx%04i.fits""",
      },
 
 'isochrone': {
     'mass_init_field': 'M_ini',
     'mass_act_field': 'M_act',
     'luminosity_field': 'logL/Lo',
     'mag_1_field': 'g',
     'mag_2_field': 'r',
     'stage_field': 'stage',
     'imf': 'chabrier',
               #'infiles': ['/home/s1/bechtol/des10.a/projects/mw_substructure/stellar_evolution/isochrones/des/isota1010z0.1.dat',
               #            '/home/s1/bechtol/des10.a/projects/mw_substructure/stellar_evolution/isochrones/des/isota1010z0.2.dat',
               #            '/home/s1/bechtol/des10.a/projects/mw_substructure/stellar_evolution/isochrones/des/isota1015z0.1.dat',
               #            '/home/s1/bechtol/des10.a/projects/mw_substructure/stellar_evolution/isochrones/des/isota1015z0.2.dat'],
     #fe/h = [-2.27,-1.5]; z    = [0.10 , 0.6]
     #age  = [8, 14]     ; age* = [090., 114.00]
     'infiles': ['/u/ki/kadrlica/sdss/data/isochrones/isota90.00z0.120.dat',
                 '/u/ki/kadrlica/sdss/data/isochrones/isota90.00z0.600.dat',
                 '/u/ki/kadrlica/sdss/data/isochrones/isota110.00z0.120.dat',
                 '/u/ki/kadrlica/sdss/data/isochrones/isota110.00z0.600.dat'],
               'weights': [0.25, 0.25, 0.25, 0.25],
               'mass_steps': 400},

 'likelihood': {
     'interior_roi': True,
     'distance_modulus_array': [19., 20.],
     'full_pdf': False,
     #'color_lut_infile': '/u/ki/kadrlica/sdss/data/isochrones/color_lut_full.fits',
     'color_lut_infile': None,
     'color_lut_delta_mag': 0.03,
     'color_lut_mag_err_array': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04],
     }
}
