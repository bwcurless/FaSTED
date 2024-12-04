from __future__ import print_function
import sys

import numpy as np
import csv


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)



def fvecs_test_dim(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    eprint(dim)

def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv



data = fvecs_read("tiny5m_base.fvecs")

# fvecs_test_dim("tiny5m_base.fvecs")

dims=384
rows=(int)(data.size*1.0/dims*1.0)
maxval=np.max(data)

# eprint("maxval: %f" %(maxval))
# eprint("Rows: %d" % (rows))
# eprint("Dims: %d" % (dims))




#print to stdout normalized or unnormalized data

normalized=0


if(normalized==0):
    for x in range (0, rows):
            for y in range (0, dims):
                    print("%f, " %(data[x][y]), end="")
            print("")

if(normalized==1):
	for x in range (0, rows):
        	for y in range (0, dims):
                	print("%f, " %(data[x][y]/maxval), end="")
        	print("")

exit(0)


#sanity check

#sanity check normalized data
#don't store data, just compute the min/max because the data is large and it runs out of memory

sanitycheck=1




if (sanitycheck):
    minval=100.0
    maxval=0.0

    tmp1=[]
    tmp1.append(0)
    tmp1.append(1)

    tmp2=[]
    tmp2.append(2)
    tmp2.append(3)

    tmp1=np.asfarray(tmp1)
    tmp2=np.asfarray(tmp2)
    datanormalized=[]
    with open('tiny5m_normalized_0_1.txt', newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                    # datanormalized.append(row[:-1]
                    # minvalinrow=np.asfarray(row[:-1])
                    tmp1[0]=np.min(np.asfarray(row[:-1]))
                    tmp1[1]=minval
                    minval=min(tmp1)

                    tmp2[0]=np.max(np.asfarray(row[:-1]))
                    tmp2[1]=maxval
                    maxval=max(tmp2)


    print(minval)
    print(maxval)

