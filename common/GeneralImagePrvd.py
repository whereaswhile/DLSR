# Regression data provider for processing any general single image

import os
import sys
import numpy as np
import scipy.misc
import scipy.io
import cPickle as pickle
import glob
sys.path.append("../convnet-folk_master")
from w_util import readLines

import cv2

class ListaSet:
    def __init__(self, paramfile):
        print "Here is in " + os.path.basename(__file__)
        self.param = {'paramfile': paramfile}
        plines=readLines(paramfile)
        for l in plines:
            l=l.rstrip().split()
            self.param[l[0]]=l[1]
               
        d=scipy.io.loadmat(self.param['imgdata'])
        self.input = []
        self.output = []

        if self.param['train']=='1':
            # self.datanum =  140*64 # 140*64
            self.datanum = d['Input'].shape[1]
        else:
            # self.datanum =  193 # 140*64
            self.datanum = d['Input'].shape[1]
        
        for i in range( self.datanum):
            self.input.append( d['Input'][0,i])
            self.output.append( d['Output'][0,i])
            
        self.indim = np.prod( np.shape(d['Input'][0,i]))
        self.outdim = np.prod( np.shape(d['Output'][0,i])) # 3rd dim is 13!
        
        print 'Length of self.input = ', len( self.input)
        if self.param['train']=='0':
            self.output=[_*0 for _ in self.output]
            
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


