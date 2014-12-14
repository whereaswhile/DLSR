# LISTA algorithm regression data provider

import os
import sys
import numpy as np
from numpy.random import RandomState
import scipy.misc
import scipy.io
import cPickle as pickle
import glob
sys.path.append("../convnet-folk_master")
from w_util import readLines

class ListaSet:
    def __init__(self, paramfile):
        self.param = {'paramfile': paramfile}
        plines=readLines(paramfile)
        for l in plines:
            l=l.rstrip().split()
            self.param[l[0]]=l[1]
        self.param['outsize']=int(self.param['outsize'])
 	print self.param
        print "ListaPrvd_full with paramfile:", paramfile

        self.indim=self.param['outsize']**2*16
        self.outdim=self.param['outsize']**2
        self.datanum=2
        self.prng = RandomState(1234567890)
        print '%d samples found' % self.datanum

    def get_num_images(self):
        return self.datanum
    
    def get_input_dim(self):
        return self.indim

    def get_output_dim(self):
        return self.outdim

    def get_input(self, idx):
        #print 'get_input!'
        res=np.zeros((self.param['outsize'], self.param['outsize'], 16), dtype=float)
        res[:,:,0:2]=(self.prng.rand(self.param['outsize'], self.param['outsize'], 2)-0.5)/10+1
        res[:,:,2:]=(self.prng.rand(self.param['outsize'], self.param['outsize'], 14)-0.5)+1
        return res

    def get_output(self, idx):
        res=np.zeros((self.param['outsize'], self.param['outsize']), dtype=float)+1
        return res
    
    def getmeta(self, idx):
        return self.param

def getStore(param):
    return ListaSet(param)

def test(param):
    ts = ListaSet(param)
    print "{} images in total".format(ts.get_num_images())
    for i in range(1,100,10):
        im=ts.get_input(i)
        y=ts.get_output(i)
        print "i={}, input={},\toutput={}".format(i, im.shape, y.shape)
        print im
        print y
    print 'image shape:', np.shape(im)

if __name__ == '__main__':
    print 'testing Toy ListaPrvd.py!'
    assert(len(sys.argv)==2)
    test(sys.argv[1])


