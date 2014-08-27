# LISTA algorithm regression data provider

import os
import sys
import numpy as np
import scipy.misc
import scipy.io
import cPickle as pickle
import glob
sys.path.append("../convnet-folk_master")
from w_util import readLines

class ListaSet:
    def __init__(self, paramfile):
        MAT_IN_VAR='Input_SR' # define default parameters
        MAT_OUT_VAR='hIm1' #'Z'

        self.param = {'paramfile': paramfile}
        plines=readLines(paramfile)
        for l in plines:
            l=l.rstrip().split()
            self.param[l[0]]=l[1]
# 	print self.param
        if 'MAT_IN_VAR' in self.param:
            MAT_IN_VAR=self.param['MAT_IN_VAR']
        if 'MAT_OUT_VAR' in self.param:
            MAT_OUT_VAR=self.param['MAT_OUT_VAR']
        print "ListaPrvd_full with paramfile:", paramfile, " (", MAT_IN_VAR, MAT_OUT_VAR, ")"

        d=scipy.io.loadmat(self.param['imgdata'])
        self.X=d[MAT_IN_VAR]
        self.Z=d[MAT_OUT_VAR]
        assert(len(np.shape(self.X))==2)
        if np.shape(self.X)[0]!=1: #single sample
            self.datanum=1 # data number
            self.indim=np.prod(np.shape(self.X))
            self.outdim=np.prod(np.shape(self.Z))
            self.input=[self.X]
            self.output=[self.Z]
        else:
            self.datanum=self.X.shape[1]
            self.indim=np.prod(np.shape(self.X[0,0]))
            self.outdim=np.prod(np.shape(self.Z[0,0]))
            self.input=[self.X[0, i] for i in range(self.datanum)]
            self.output=[self.Z[0, i] for i in range(self.datanum)]
        if self.param['train']=='0':
            self.output=[_*0 for _ in self.output]
        print '%d samples found' % self.datanum

    def get_num_images(self):
        return self.datanum
    
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
    return ListaSet(param)


def test(param):
    ts = ListaSet(param)
    print "{} images in total".format(ts.get_num_images())
    for i in range(0,20,10):
        im=ts.get_input(i)
        y=ts.get_output(i)
        meta=ts.getmeta(i)
        print "i={}, input={},\toutput={}".format(i, im, y)
    print 'image shape:', np.shape(im)
    print 'meta', meta

if __name__ == '__main__':
    print 'testing ListaPrvd.py!'
    assert(len(sys.argv)==2)
    test(sys.argv[1])


