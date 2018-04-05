#!/usr/bin/env python
from os.path import join
import pylab as plt
import pywcsgrid2
import yaml
import matplotlib.patheffects as patheffects

import numpy as np
import fitsio
import astropy.io.fits as pyfits

from astropy.coordinates import SkyCoord
import astropy.units as u

COLORS = dict(gal='0.5',fk5='0.0')

effects=[patheffects.withStroke(linewidth=3,foreground="w")]
                                          
def create_header(coord, radius, proj='ZEA', npix=30):
    """ Create a header a new image """
    gal = coord.name == 'galactic'
    values = [
            ["NAXIS",  2,          ],
 
            ["NAXIS1", npix,       ],
            ["NAXIS2", npix,       ],
 
            ["CTYPE1", 'GLON-%s'%proj if gal else 'RA---%s'%proj ],
            ["CTYPE2", 'GLAT-%s'%proj if gal else 'DEC--%s'%proj ],
 
            ["CRPIX1", npix/2. + 0.5,       ],
            ["CRPIX2", npix/2. + 0.5,       ],
 
            ["CRVAL1", coord.l.deg if gal else coord.ra.deg,        ],
            ["CRVAL2", coord.b.deg if gal else coord.dec.deg,       ],
 
            ["CDELT1", -3.*radius/npix,       ],
            ["CDELT2",  3.*radius/npix,        ],
    ]
 
    if not gal:
        values += [
            ['RADECSYS','FK5'],
            ['EQUINOX',2000],
        ]
 
    cards = [pyfits.Card(*i) for i in values]
    header=pyfits.Header(cards=cards)
 
    return header

def create_header_data(coord, radius=10., **kwargs):
    """ Make an empty sky region at location of skydir
    skydir : skymaps.SkyDir object
    size   : size of region (deg.)
    kwargs : arguments passed to create_header
    """
    header = create_header(coord, radius=radius, **kwargs)
    data = np.zeros( (header['NAXIS1'],header['NAXIS2']) )
    return header, data


def plot_skyreg(header, data, **kwargs):
    """ Plot sky region defined by header and data
    header : FITS header
    data   : Data array
    """
    kwargs.setdefault('cmap','binary')
    fig = plt.figure()
    ax = pywcsgrid2.subplot(111, header=header)
    ax.set_ticklabel_type("dms")
    im = ax.imshow(data, origin="center", **kwargs)
    ax.grid()
    ax.add_compass(loc=1,coord='fk5')
    ax.add_compass(loc=4,coord='gal')
    return ax, im


def draw_scatter(coord, data, **kwargs):
    kwargs.setdefault('s',20)
    kwargs.setdefault('edgecolor','none')
    kwargs.setdefault('vmin',0)
    kwargs.setdefault('vmax',1)
    kwargs.setdefault('zorder',3)

    bkg_kwargs = dict(kwargs)
    bkg_kwargs['s'] = 5
    bkg_kwargs['zorder'] = 2
    bkg_kwargs['c'] = '0.70'

    try: 
        data =  data[np.argsort(data['PROB'])]
    except:
        pass 

    prob = data['PROB']
    cut = (prob > 0.05)

    sys = 'gal' if (coord.name == 'galactic') else 'fk5'
    if sys == 'gal':
        lon,lat = data['GLON'],data['GLAT']
    else:
        lon,lat = data['RA'],data['DEC']

    print(len(cut), cut.sum(), (cut==False).sum())
    
    ax[sys].scatter(lon[~cut],lat[~cut],**bkg_kwargs)
    ax[sys].scatter(lon[cut],lat[cut],c=prob[cut],**kwargs)


def delta_coord(coord,angle,offset=1e-4):
    angle_deg = angle*np.pi/180
    newlat = offset * np.cos(angle_deg) + coord.data.lat.degree
    newlon = (offset * np.sin(angle_deg) / np.cos(newlat * np.pi/180) + coord.data.lon.degree)
    return SkyCoord(newlon, newlat, unit='degree', frame=coord.frame.name)

def estimate_angle(coord, angle, new_frame, offset=1e-7):
    """
    https://github.com/astropy/astropy/issues/3093
    """
    delta = delta_coord(coord, angle, offset)
    new_coord = coord.transform_to(new_frame)
    new_delta = delta.transform_to(new_frame)
    return new_coord.position_angle(new_delta).deg

def gal2cel_angle(glon,glat,angle,offset=1e-7):
    coord = SkyCoord(glon,glat,unit=u.deg,frame='galactic')
    return estimate_angle(coord,angle,'fk5',offset)

def cel2gal_angle(ra,dec,angle,offset=1e-7):
    coord = SkyCoord(ra,dec,unit=u.deg,frame='fk5')
    return estimate_angle(coord,angle,'galactic',offset)


if __name__ == "__main__":
    import ugali.utils.parser
    description = "python script"
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_coords(required=True,targets=True)
    parser.add_name()
    opts = parser.parse_args()

    dirname = 'mcmc_v08'
    for name,c in zip(opts.names,opts.coords):
        if (opts.name is not None) and name != opts.name: continue
        print(name,c)

        memfile = join(dirname,'%s_mcmc.fits'%name)
        resfile = join(dirname,'%s_mcmc.yaml'%name)

        coord = SkyCoord(c[0]*u.deg,c[1]*u.deg,frame='galactic')
        glon,glat = coord.galactic.l.deg,coord.galactic.b.deg
        ra,dec = coord.fk5.ra.deg,coord.fk5.dec.deg

        data = pyfits.open(memfile)[1].data
        params  = yaml.load(open(resfile))['params']
        results = yaml.load(open(resfile))['results']
        pa_gal = params['position_angle']

        pa_cel = gal2cel_angle(coord.l.deg,coord.b.deg,pa_gal) % 180
        
        for frame in ['galactic','fk5']:
            co = getattr(coord,frame)

            h,d = create_header_data(co,radius=0.4)
            ax, im = plot_skyreg(h,d)
            plt.sca(ax)
            draw_scatter(co,data)

            d_gal = delta_coord(co.galactic,pa_gal,0.15)
            d_fk5 = delta_coord(co.fk5,pa_cel,0.15)

            ax['gal'].annotate('',xy=(d_gal.l.deg,d_gal.b.deg),
                               xytext=(glon,glat),textcoords='data',
                               arrowprops=dict(arrowstyle='->',color=COLORS['gal']),zorder=10)
            ax['fk5'].annotate('',xy=(d_fk5.ra.deg,d_fk5.dec.deg),
                               xytext=(ra,dec),textcoords='data',
                               arrowprops=dict(arrowstyle='->',color=COLORS['fk5']),zorder=10)

            plt.plot(np.nan,np.nan,'-',c=COLORS['gal'],label=r'${\rm PA_{GAL}} = %.0f$'%pa_gal)
            plt.plot(np.nan,np.nan,'-',c=COLORS['fk5'],label=r'${\rm PA_{CEL}} = %.0f$'%pa_cel)
            plt.legend(loc=2,fontsize=10)

            plt.annotate(r'$\ell,b = %.2f,%.2f$'%(glon,glat),xy=(0.1,0.1),xycoords='axes fraction',fontsize=10)
            plt.annotate(r'${\rm RA,DEC} = %.2f,%.2f$'%(ra,dec),xy=(0.1,0.15),xycoords='axes fraction',fontsize=10)

            #plt.annotate(r'${\rm PA_{GAL}} = %.0f$'%pa_gal,xy=(0.05,0.95),xycoords="axes fraction",fontsize=10)
            #plt.annotate(r'${\rm PA_{CEL}} = %.0f$'%pa_cel,xy=(0.05,0.90),xycoords="axes fraction",fontsize=10)
            #plt.annotate(r'${\rm PA_{CEL}} = %.0f$'%(pa_cel-180),xy=(0.05,0.85),xycoords="axes fraction",fontsize=10)

            ax.set_title('%s (%s)'%(name,frame[:3]))
            plt.savefig('%s_pa_%s.png'%(name,frame[:3]),bbox_inches='tight')
            break
    plt.ion()
    plt.show()
