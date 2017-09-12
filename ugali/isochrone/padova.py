#!/usr/bin/env python
"""
Older (pre-PARSEC) Padova isochrones.
"""

class Girardi2002(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'gi2000'

class Marigo2008(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'ma08'

class Girardi2010a(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'gi10a'

class Girardi2010b(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'gi10b'

class Girardi2002(PadovaIsochrone):
    #_dirname = '/u/ki/kadrlica/des/isochrones/v5/'
    _dirname =  os.path.join(get_iso_dir(),'{survey}','girardi2002')
    # For use with Marigo et al. (2008) and earlier use Anders & Grevesse 1989
    _zsolar = 0.019

    defaults = (Isochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
        ('hb_stage','BHeb','Horizontal branch stage name'),
        ('hb_spread',0.1,'Intrinisic spread added to horizontal branch'),
        )

    columns = dict(
        des = odict([
                (2, ('mass_init',float)),
                (3, ('mass_act',float)),
                (4, ('log_lum',float)),
                (9, ('g',float)),
                (10, ('r',float)),
                (11,('i',float)),
                (12,('z',float)),
                (13,('Y',float)),
                (15,('stage',object))
                ]),
        )
    
    def _parse(self,filename):
        """
        Reads an isochrone in the old Padova format (Girardi 2002,
        Marigo 2008) and determines the age (log10 yrs and Gyr),
        metallicity (Z and [Fe/H]), and creates arrays with the
        initial stellar mass and corresponding magnitudes for each
        step along the isochrone.
        http://stev.oapd.inaf.it/cgi-bin/cmd
        """
        try:
            columns = self.columns[self.survey.lower()]
        except KeyError, e:
            logger.warning('did not recognize survey %s'%(survey))
            raise(e)

        kwargs = dict(delimiter='\t',usecols=columns.keys(),dtype=columns.values())
        self.data = np.genfromtxt(filename,**kwargs)
        
        self.mass_init = self.data['mass_init']
        self.mass_act  = self.data['mass_act']
        self.luminosity = 10**self.data['log_lum']
        self.mag_1 = self.data[self.band_1]
        self.mag_2 = self.data[self.band_2]
        self.stage = np.char.array(self.data['stage']).strip()
        for i,s in enumerate(self.stage):
            if i>0 and s=='' and self.stage[i-1]!='':
                self.stage[i] = self.stage[i-1]

        # Check where post-AGB isochrone data points begin
        self.mass_init_upper_bound = np.max(self.mass_init)
        if np.any(self.stage == 'LTP'):
            self.index = np.nonzero(self.stage == 'LTP')[0][0]
        else:
            self.index = len(self.mass_init)

        self.mag = self.mag_1 if self.band_1_detection else self.mag_2
        self.color = self.mag_1 - self.mag_2

class Girardi2010(Girardi2002):
    #_dirname = '/u/ki/kadrlica/des/isochrones/v4/'
    _dirname =  os.path.join(get_iso_dir(),'{survey}','girardi2010')
    _zsolar = 0.019

    defaults = (Isochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
        ('hb_stage','BHeb','Horizontal branch stage name'),
        ('hb_spread',0.1,'Intrinisic spread added to horizontal branch'),
        )

    columns = dict(
        des = odict([
                (2, ('mass_init',float)),
                (3, ('mass_act',float)),
                (4, ('log_lum',float)),
                (9, ('g',float)),
                (10,('r',float)),
                (11,('i',float)),
                (12,('z',float)),
                (13,('Y',float)),
                (19,('stage',object))
                ]),
        )
