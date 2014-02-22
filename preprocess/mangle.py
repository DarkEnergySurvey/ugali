#!/usr/bin/env python
import os, sys
from os.path import splitext, exists
import subprocess

def fits2ply(infile,outfile=None):
    base,ext = os.path.splitext(infile)
    if ext != '.fits': 
        raise Exception("Infile must be FITS.")

    if outfile is None:
        outfile = base + '.ply'
    if os.path.exists(outfile): os.remove(outfile)

    print "Converting %s to %s"%(infile,outfile)

    ### Python Routine
    import mangle
    mask = mangle.Mangle(infile)
    mask.write(outfile)

    #### IDL Routine
    ## Should use a tmp file...
    #profile = 'fits2ply.pro'
    #if os.path.exists(profile): os.remove(profile)
    #pro = open(profile,'w')
    #pro.write("""
    #pro fits2ply
    #  args = command_line_args()
    #  infile = args[0]
    #  outfile = args[1]
    # 
    #  read_fits_polygons, infile, polygons
    #  write_mangle_polygons, outfile, polygons
    #end"""
    #)
    #pro.close()
    # 
    #cmd = 'idl -e "fits2ply" -args %s %s'%(infile, outfile)
    #print cmd
    #subprocess.call(cmd, shell=True)
    #os.remove(profile)

def poly2poly(infile, outfile, *opts):
    opts = ' '.join(opts)
    cmd = "poly2poly %s %s %s"%(opts,infile,outfile)
    print cmd
    subprocess.call(cmd,shell=True)

def pixelize(infile, outfile, *opts):
    opts = ' '.join(opts)
    cmd = "pixelize %s %s %s"%(opts,infile,outfile)
    print cmd
    subprocess.call(cmd,shell=True)

def snap(infile, outfile, *opts):
    opts = ' '.join(opts)
    cmd = "snap %s %s %s"%(opts,infile,outfile)
    print cmd
    subprocess.call(cmd,shell=True)
    
def balkanize(infile, outfile, *opts):
    opts = ' '.join(opts)
    cmd = "balkanize %s %s %s"%(opts,infile,outfile)
    print cmd
    subprocess.call(cmd,shell=True)
        
def unify(infile, outfile, *opts):
    opts = ' '.join(opts)
    cmd = "unify %s %s %s"%(balkfile,unifile,opts)
    print cmd
    subprocess.call(cmd,shell=True)    

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    parser.add_option("-p","--pix",default=8, type='int',
                      help="Mangle pixelization")
    parser.add_option("-s","--snap",default=0.1, type='float',
                      help="Snap tolerance.")
    (opts, args) = parser.parse_args()

    if not len(args):
        print "ERROR: Infile require."
        parser.print_help()
        sys.exit(1)
     
    infile = args[0]
    base, ext = os.path.splitext(infile)
    #windows = ['window.boss.dr9.ply']
    #masks   = ['mask.boss.dr9.field.ply','mask.boss.dr9.star.ply']
    #holes   = []
    #base    = splitext(windows[0])[0]
    ## This is to match up with the HEALPix polygons
    #if   nside > 32: pix = 8
    #elif nside < 16: pix = 3
    #else:            pix = 5
        
    ## Polygonize
    #if ext == ".fits":
    #    polyfile = base+'.ply'
    #    cmd = "fits2ply %s %s"%(infile,polyfile)
    #    print cmd
    #    subprocess.call(cmd,shell=True)
    #else:
    #    polyfile = infile
    #if masks:
    #    jw = open('jw','w')
    #    jw.write('0\n')
    #    jw.close()
    # 
    #for mask in masks:
    #    hole = mask.replace('mask.','holes.')
    #    if not os.path.exists(hole):
    #        cmd = "weight -zjw %s %s"%(mask,hole)
    #        print cmd
    #        subprocess.call(cmd,shell=True)
    #    holes.append(hole)

    # To create the holy file
    # ???
    
    snapopts= "-a.1 -b.1 -t.1"
    pixopts = "-Ps0,%i"%opts.pix
    mtol="-m1e-8"
    polyopts = '-k1e-11'

    # Poly2Poly
    #polyfile = base + '.poly.ply'
    #if not os.path.exists(polyfile):
    #    poly2poly(infile,polyfile,polyopts,mtol)
    #else: print "Found %s; skipping..."%polyfile
    # 
    ## Pre-snap (only self-snap)
    #snapfile =  base + '.psnap.ply'
    #if not os.path.exists(snapfile):
    #    snap(infile,snapfile,"-S ",snapopts,mtol)
    #else: print "Found %s; skipping..."%snapfile
    # 
    ## Pixelize
    pixfile = base + '.pix.ply'
    if not os.path.exists(pixfile):
        pixelize(infile,pixfile,pixopts,mtol)
    else: print "Found %s; skipping..."%pixfile

    # Snap
    snapfile =  base + '.snap.ply'
    if not os.path.exists(snapfile):
        snap(pixfile,snapfile,snapopts,mtol)
    else: print "Found %s; skipping..."%snapfile

    # Balkanize
    balkfile = base + '.balk.ply'
    if not os.path.exists(balkfile):
        balkanize(snapfile,balkfile,mtol)
    else: print "Found %s; skipping..."%balkfile

    # Unify
    unifile = base + '.uni.ply'
    if not os.path.exists(unifile):
        unify(balkfile, unifile)
    else: print "Found %s; skipping..."%unifile

