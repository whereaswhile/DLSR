# LISTA algorithm regression data provider
# obtain samples by online sampling from images

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
        self.param = {'paramfile': paramfile, 'rotate': 1}
        plines=readLines(paramfile)
        for l in plines:
            l=l.rstrip().split()
            self.param[l[0]]=l[1]
        self.param['inpsize']=int(self.param['inpsize'])
        self.param['outsize']=int(self.param['outsize'])
        self.param['mrgsize']=(self.param['inpsize']-self.param['outsize'])/2
        self.param['smplside']=int(self.param['smplPerSide'])
        self.param['rotate']=max(1, min(4, int(self.param['rotate'])))
# 	print self.param
        MAT_IN_VAR=self.param['MAT_IN_VAR']
        MAT_OUT_VAR=self.param['MAT_OUT_VAR']
        print "ListaPrvd_full with paramfile:", paramfile, " (", MAT_IN_VAR, MAT_OUT_VAR, ")"

        d=scipy.io.loadmat(self.param['imgdata'])
        self.X=d[MAT_IN_VAR]
        self.Z=d[MAT_OUT_VAR]
        assert(len(np.shape(self.X))==2)
        self.imgnum=self.X.shape[1]
        self.pertnum=(self.param['smplside']**2)*self.param['rotate']
        self.datanum=self.imgnum*self.pertnum
        self.indim=self.param['inpsize']**2 #np.prod(np.shape(self.X[0,0]))
        self.outdim=self.param['outsize']**2 #np.prod(np.shape(self.Z[0,0]))
        self.input=[self.X[0, i] for i in range(self.imgnum)]
        self.output=[self.Z[0, i] for i in range(self.imgnum)]
        if self.param['train']=='0':
            self.output=[_*0 for _ in self.output]
        print '%d images, %d samples found' % (self.imgnum, self.datanum)

    def get_num_images(self):
        return self.datanum
    
    def get_input_dim(self):
        return self.indim

    def get_output_dim(self):
        return self.outdim

    def get_input(self, idx):
        img_id=idx/self.pertnum
        pert_id=idx%self.pertnum
        rot_id=pert_id%self.param['rotate']
        off_id=pert_id/self.param['rotate']
        [h, w]=self.input[img_id].shape
        [dy, dx]=self.get_offset(h, w, off_id)
        res=self.input[img_id][dy:dy+self.param['inpsize'], dx:dx+self.param['inpsize']]
        #res=np.rot90(res) #rotate 90
        if rot_id==1:
            res=np.fliplr(res)
        elif rot_id==2:
            res=np.flipud(res)
        elif rot_id==3:
            res=res.T
        return res

    def get_output(self, idx):
        img_id=idx/self.pertnum
        pert_id=idx%self.pertnum
        rot_id=pert_id%self.param['rotate']
        off_id=pert_id/self.param['rotate']
        [h, w]=self.output[img_id].shape
        [dy, dx]=self.get_offset(h, w, off_id)
        dy+=self.param['mrgsize']
        dx+=self.param['mrgsize']
        res=self.output[img_id][dy:dy+self.param['outsize'], dx:dx+self.param['outsize']]
        #res=np.rot90(res) #rotate 90
        if rot_id==1:
            res=np.fliplr(res)
        elif rot_id==2:
            res=np.flipud(res)
        elif rot_id==3:
            res=res.T
        return res
    
    def getmeta(self, idx):
        return self.param
                
    def get_offset(self, h, w, i):
        iy=i/self.param['smplside']
        ix=i%self.param['smplside']
        dy=iy*((h-self.param['inpsize'])/(self.param['smplside']-1))
        dx=ix*((w-self.param['inpsize'])/(self.param['smplside']-1))
        return dy, dx

def getStore(param):
    return ListaSet(param)


def test(param):
    ts = ListaSet(param)
    print "{} images in total".format(ts.get_num_images())
    for i in range(728,748,1):
        im=ts.get_input(i)
        y=ts.get_output(i)
        print "i={}, input={},\toutput={}".format(i, im.shape, y.shape)
        scipy.misc.imsave('./img/{}_in.png'.format(i), im);
        scipy.misc.imsave('./img/{}_out.png'.format(i), y);
    print 'image shape:', np.shape(im)

if __name__ == '__main__':
    print 'testing ListaPrvd.py!'
    assert(len(sys.argv)==2)
    test(sys.argv[1])


