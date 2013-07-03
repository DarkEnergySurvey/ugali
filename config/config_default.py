{'output': {'chatter': 4, # 0 = silent; 1 = text output; 2 = more text output; 3 = text output and figures; 4 = paranoid
            'savedir_mag_1_mask': '/home/s1/bechtol/des10.a/projects/mw_substructure/sv_test/bullet/mask_r',
            'savedir_mag_2_mask': '/home/s1/bechtol/des10.a/projects/mw_substructure/sv_test/bullet/mask_i',
            'savedir_likelihood': '/home/s1/bechtol/des10.a/projects/mw_substructure/sv_test/bullet/likelihood'},

 'catalog': {'infile': '/home/s1/bechtol/des10.a/projects/mw_substructure/mangle_test/y1c1_coadd_BulletClustersmask/data/starcat_y1c1_coadd_bullet.fits',
             #'infile': '/home/s1/bechtol/des10.a/projects/dust_maps/cluster_fields/bullet/image_riz_starcat_slr.fits',
             'lon_field': 'RA',
             'lat_field': 'DEC',
             'coordsys': 'cel',
             'mag_1_field': 'MAG_PSF_R', # Color is defined as mag_1 - mag_2
             'mag_err_1_field': 'MAGERR_PSF_R',
             'mag_2_field': 'MAG_PSF_I',
             'mag_err_2_field': 'MAGERR_PSF_I',
             #'mc_source_id_field': 'mc_source_id',
             'mc_source_id_field': None,
             'band_1_detection': False}, # True = band 1 is detection band; False = band 2 is detection band

 'mangle': {'infile_1': '/home/s1/bechtol/des10.a/projects/mw_substructure/mangle_test/y1c1_coadd_BulletClustersmask/y1c1_coadd_BulletClusters_holymolys_maglims_r.pol',
            'infile_2': '/home/s1/bechtol/des10.a/projects/mw_substructure/mangle_test/y1c1_coadd_BulletClustersmask/y1c1_coadd_BulletClusters_holymolys_maglims_i.pol',
            'coordsys': 'cel'},
 #'manglebindir': '/home/s1/bechtol/des10.a/code/mangle/mangle2.2/bin'},

 'mask': {'infile_1': '/home/s1/bechtol/des10.a/projects/mw_substructure/sv_test/bullet/mask_r.fits',
          'infile_2': '/home/s1/bechtol/des10.a/projects/mw_substructure/sv_test/bullet/mask_i.fits',
          'minimum_solid_angle': 0.1}, # deg^2
          
 'color': {'min': -0.5,
           'max': 1., # 1.5
           'n_bins': 15}, # 0.1, 0.02

 'mag': {'min': 18,
         'max': 25, # 25
         'n_bins': 70}, # 0.1, 0.02

 # HEALPix nside choices
 # 2^6  -> 55.0 arcmin,   0.84 deg^2, 3021 arcmin^2
 # 2^7  -> 27.5 arcmin,   0.21 deg^2,  755 arcmin^2
 # 2^8  -> 13.7 arcmin,  0.052 deg^2,  189 arcmin^2
 # 2^9  -> 6.87 arcmin,  0.013 deg^2, 47.2 arcmin^2
 # 2^10 -> 3.44 arcmin, 3.3e-3 deg^2, 11.8 arcmin^2
 # 2^11 -> 1.72 arcmin, 8.2e-4 deg^2, 2.95 arcmin^2
 # 2^12 -> 0.86 arcmin, 2.0e-4 deg^2, 0.74 arcmin^2
 # 2^13 -> 0.43 arcmin, 5.1e-5 deg^2, 0.18 arcmin^2

 'coords': {'nside_likelihood_segmentation': 2**8, # Size of patches for likelihood analysis
            'nside_mask_segmentation': 2**6, # Size of patches for mask creation
            'nside_pixel': 2**12, 
            'roi_radius': 1.,
            'coordsys': 'gal',
            'proj_type': 'ait'},

 'kernel': {'type': 'plummer',
            'params': [0.1]},
 
 'isochrone': {'dir': '/home/s1/bechtol/des10.a/projects/mw_substructure/stellar_evolution/isochrones/des/',
               'mass_init_field': 'M_ini',
               'mass_act_field': 'M_act',
               'mag_1_field': 'r',
               'mag_2_field': 'i',
               'stage_field': 'stage',
               'imf': 'chabrier',
               'mass_steps': 400},

 'likelihood': {'distance_modulus_array': [18., 19.]}
 }
