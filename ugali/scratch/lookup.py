#!/usr/bin/env python
import ugali.candidate.associate
import ugali.utils.parser
import numpy as np
from ugali.utils.projector import gal2cel,ang2const,ang2iau

#CATALOGS = ['McConnachie12','Rykoff14', 'Harris96', 'Corwen04', 'Nilson73', 'Webbink85', 'Kharchenko13', 'WEBDA14']
CATALOGS = ['McConnachie12', 'Harris96', 'Corwen04', 'Nilson73', 'Webbink85', 'Kharchenko13', 'Bica08', 'WEBDA14', 'ExtraDwarfs','ExtraClusters']

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_coords(required=True,radius=True,targets=True)
    parser.add_argument('-n','--nnearest',default=1,type=int)
    opts = parser.parse_args()

    catalog = ugali.candidate.associate.SourceCatalog()
    for i in CATALOGS:
        catalog += ugali.candidate.associate.catalogFactory(i)

    for name,(glon,glat,radius) in zip(opts.names, opts.coords):
        ra,dec = gal2cel(glon,glat)
        iau = ang2iau(glon,glat)
        const = ang2const(glon,glat)[0]
        if radius <= 0: radius = None
    
        idx1,idx2,sep = catalog.match([glon],[glat],tol=radius,nnearest=opts.nnearest)
        match = catalog[idx2]

        if len(match) > 0:
            n = match[0]['name']
            s = sep[0]
            l,b = match[0]['glon'],match[0]['glat']
            r,d = match[0]['ra'],match[0]['dec']
        else:
            n = 'NONE'
            s = np.nan
            l,b = np.nan,np.nan
            r,d = np.nan,np.nan

            
        if opts.gal is not None:
            msg='%s [%s, %s] (GLON=%.2f,GLAT=%.2f) --> %s (GLON=%.2f,GLAT=%.2f): %.4f'%(name,iau,const,glon,glat,n,l,b,s)
        else:
            msg='%s [%s, %s] (RA=%.2f,DEC=%.2f) --> %s (RA=%.2f,DEC=%.2f): %.4f'%(name,iau,const,ra,dec,n,r,d,s)
        print(msg)


        #for i,c in enumerate(opts.coords):
        #    glon,glat,radius
        #    if i in idx1:
        #        name = catalog[idx2[np.where(idx1==i)[0]]]['name']
        #        s = sep[np.where(idx1==i)[0]]
        #    else:
        #        name = "NONE"
        #        s = np.nan
        # 
        #    if opts.gal is not None:
        #        msg='%s (GLON=%.2f,GLAT=%.2f): %.4f'%(name,c[1],c[2],s)
        #    else:
        #        ra,dec = gal2cel(c1,c2)
        #        msg='%s (RA=%.2f,DEC=%.2f): %.4f'%(name,ra,dec,s)
        #    print msg
