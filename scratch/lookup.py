#!/usr/bin/env python
import ugali.candidate.associate
import ugali.utils.parser

#CATALOGS = ['McConnachie12','Rykoff14', 'Harris96', 'Corwen04', 'Nilson73', 'Webbink85', 'Kharchenko13', 'WEBDA14']
CATALOGS = ['McConnachie12', 'Harris96', 'Corwen04', 'Nilson73', 'Webbink85', 'Kharchenko13', 'WEBDA14']

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


    for glon,glat,radius in opts.coords:
        if radius <= 0: radius = None
    
        idx1,idx2,sep = catalog.match([glon],[glat],tol=radius,nnearest=opts.nnearest)
        match = catalog[idx2]
         
        for i,m in enumerate(match):
            if opts.gal is not None:
                msg='%s (GLON=%.2f,GLAT=%.2f): %.4f'%(m['name'],m['glon'],m['glat'],sep[i])
            else:
                msg='%s (RA=%.2f,DEC=%.2f): %.4f'%(m['name'],m['ra'],m['dec'],sep[i])
            print msg
