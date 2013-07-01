{'output': {'chatter': 4, # 0 = silent; 1 = text output; 2 = more text output; 3 = text output and figures; 4 = paranoid
            'savedir': None}, # Output files go here
 
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

 'mask': {'infile_1': '/home/s1/bechtol/des10.a/projects/mw_substructure/mangle_test/y1c1_coadd_BulletClustersmask/y1c1_coadd_BulletClusters_holymolys_maglims_r.pol',
          'infile_2': '/home/s1/bechtol/des10.a/projects/mw_substructure/mangle_test/y1c1_coadd_BulletClustersmask/y1c1_coadd_BulletClusters_holymolys_maglims_i.pol',
          'coordsys': 'cel',
          'minimum_solid_angle': 0.1}, # deg^2
          #'manglebindir': '/home/s1/bechtol/des10.a/code/mangle/mangle2.2/bin'},

 'color': {'min': -0.5,
           'max': 1., # 1.5
           'n_bins': 15}, # 0.1, 0.02

 'mag': {'min': 18,
         'max': 25, # 25
         'n_bins': 70}, # 0.1, 0.02

 #'coords': {'reference': [104.670, -55.943], # deg
 #           'pixel_size': 0.01, # deg 0.01
 #           'n_pixels': 100, # 100
 #           'coordsys': 'gal',
 #           'proj_type': 'ait'},

 'coords': {'nside_target': 2**8, # 2^7 -> 27.5 arcmin; 2^8 -> 13.7 arcmin
            'roi_radius': 1., #deg
            'nside_pixel': 2**12, # 0.86 arcmin
            'coordsys': 'gal',
            'proj_type': 'ait'},

 'kernel': {'type': 'disk',
            'params': [0.1]},
 
 'isochrone': {'dir': '/home/s1/bechtol/des10.a/projects/mw_substructure/stellar_evolution/isochrones/des/',
               'mass_init_field': 'M_ini',
               'mass_act_field': 'M_act',
               'mag_1_field': 'r',
               'mag_2_field': 'i',
               'stage_field': 'stage',
               'imf': 'chabrier',
               'mass_steps': 400},
 }
