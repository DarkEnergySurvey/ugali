import sys
import glob
import numpy as np
import astropy.io.fits as pyfits

print sys.argv

infiles = sorted(glob.glob(sys.argv[1]))
outfile = sys.argv[2]

data_array = []
header_array = []
for infile in infiles:
    print infile
    reader = pyfits.open(infile)
    data_array.append(reader[1].data)
    header_array.append(reader[1].header)
    reader.close()

data_array = np.concatenate(data_array)


print '\nWill write output to %s\n'%(outfile)

tbhdu = pyfits.BinTableHDU(data_array)
tbhdu.header = header_array[0]

raw_input('Continue?')

tbhdu.writeto(outfile)
