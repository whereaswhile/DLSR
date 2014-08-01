#simulated regression data provider

import os
import sys
import numpy as np
import scipy.misc
import glob
sys.path.append("../convnet-folk_master")
from w_util import readLines

# define default parameters
IN_DATA_SIZE=[5, 5, 1]
OUT_DATA_SIZE=[16, 1]
DATA_NUM=1000

class SimSet:
    def __init__(self, paramfile):
        
        print "SimPrvd: parsing", paramfile
        plines = readLines(paramfile)
	self.param = {'paramfile': paramfile, 'filtype': 'avg'} 
	for l in plines:
	    l=l.rstrip().split()
	    self.param[l[0]]=l[1]
	print self.param

        self.indim=1
	for s in IN_DATA_SIZE:
	    self.indim*=s
        self.outdim=1
	for s in OUT_DATA_SIZE:
	    self.outdim*=s

	# draw data
	self.input=[]
	self.output=[]
	if self.param['filtype'][-4:]=='.fil': #load filter from file
	    fil=np.loadtxt(self.param['filtype'])
	    fil=np.reshape(fil, IN_DATA_SIZE)
	for i in range(DATA_NUM):
	    m=np.random.random(IN_DATA_SIZE) #random
	    self.input+=[m]
	    mm=np.zeros(OUT_DATA_SIZE)
	    if self.param['filtype']=='avg':
	        mm[0, 0]=np.mean(m)
	    else: 
	        mm[0, 0]=np.sum(m*fil)
	    self.output+=[mm]

    def get_num_images(self):
        return DATA_NUM
    
    #def get_num_classes(self):
    #    return 0
        
    def get_input_dim(self):
	return self.indim

    def get_output_dim(self):
	return self.outdim

    def get_input(self, idx):
        return self.input[idx]

    def get_output(self, idx):
        return self.output[idx]
    
    def getmeta(self, idx):
        return self.param
                

def getStore(param):
    return SimSet(param)


def test(param):
    ts = SimSet(param)
    print "{} images, {} classes".format(ts.get_num_images(), ts.get_num_classes())
    for i in range(0,20,10):
        im=ts.get_input(i)
	y=ts.get_output(i)
        meta=ts.getmeta(i)
        print "i={}, input={},\toutput={}".format(i, im, y)
    print 'image shape:', np.shape(im)
    print 'meta', meta

if __name__ == '__main__':
    print 'testing SimPrvd.py!'
    assert(len(sys.argv)==2)
    test(sys.argv[1])


