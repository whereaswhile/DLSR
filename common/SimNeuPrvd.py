#simulated neuron scalar data provider

import os
import sys
import numpy as np
import scipy.misc
import glob
sys.path.append("../convnet-folk_master")
from w_util import readLines

# define default parameters
DATA_NUM=1000

class SimSet:
    def __init__(self, paramfile):
        
        print "SimNeuPrvd: parsing", paramfile
        plines = readLines(paramfile)
	self.param = {'paramfile': paramfile} 
	for l in plines:
	    l=l.rstrip().split()
	    self.param[l[0]]=l[1]
	print self.param

        self.indim=1
        self.outdim=1

	# draw data
	self.input=[]
	self.output=[]
	for i in range(DATA_NUM):
	    if DATA_NUM==1:
	        m=np.ones([1])
	    else:
	        m=np.random.random(1)*2-1 #random in [-1, 1)
	    self.input+=[m]
	    mm=np.zeros([1])
	    if self.param['train']=='1':
	        #mm[0]=max(0, m[0]) #relu
	        mm[0]=max(0, abs(m[0])-0.5)*(1 if m[0]>=0 else -1) #shlu
	    else:
	        mm[0]=0
	    self.output+=[mm]

    def get_num_images(self):
        return DATA_NUM
    
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


