# image filtering regression data provider

import os
import sys
import numpy as np
import scipy.misc
import cPickle as pickle
import glob
sys.path.append("../convnet-folk_master")
from w_util import readLines

# define default parameters
IN_DATA_SIZE=[128, 128, 1]
OUT_DATA_SIZE=[16, 124*124]
DATA_NUM=1

class ImgFltrSet:
    def __init__(self, paramfile):
        
        print "ImgFltrPrvd with paramfile:", paramfile
	self.param = {'paramfile': paramfile}
        plines=readLines(paramfile)
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

	# prepare data
	d=pickle.load(open(self.param['imgdata'], "rb"))
	self.input=[np.reshape(d['img'].astype('float32'), IN_DATA_SIZE)]
	mm=np.zeros(OUT_DATA_SIZE)
	if self.param['train']=='1': #zero output for test
	    mm[0, :]=np.reshape(d['res'], [OUT_DATA_SIZE[1]])
	self.output=[mm.astype('float32')]

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
    return ImgFltrSet(param)


def test(param):
    ts = ImgFltrSet(param)
    print "{} images in total".format(ts.get_num_images())
    for i in range(0,20,10):
        im=ts.get_input(i)
	y=ts.get_output(i)
        meta=ts.getmeta(i)
        print "i={}, input={},\toutput={}".format(i, im, y)
    print 'image shape:', np.shape(im)
    print 'meta', meta

if __name__ == '__main__':
    print 'testing ImgFltrPrvd.py!'
    assert(len(sys.argv)==2)
    test(sys.argv[1])


