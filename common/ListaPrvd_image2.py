# LISTA algorithm regression data provider
# obtain samples by online sampling from images
# get the samples that has the variance larger than 10!
# Before using it, make sure line 167 & 170 in imgdata.py are uncommented!
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

        basename = os.path.basename(__file__)
        print os.path.splitext(basename)[0], " with paramfile:", paramfile, " (", MAT_IN_VAR, MAT_OUT_VAR, ")"
#        print "ListaPrvd_full with paramfile:", paramfile, " (", MAT_IN_VAR, MAT_OUT_VAR, ")"

        d=scipy.io.loadmat(self.param['imgdata'])
        self.X=d[MAT_IN_VAR]
        self.Z=d[MAT_OUT_VAR]
        assert(len(np.shape(self.X))==2)
        self.imgnum=self.X.shape[1]
        self.pertnum=(self.param['smplside']**2)*self.param['rotate']

        if 'idxdata' in self.param:
            print "'idxdata' is specified!"
            d2 = scipy.io.loadmat(self.param['idxdata'])
            self.datanum = d2['Idx_list'].shape[1]
            self.idx_arr = [d2['Idx_list'][0,i] for i in range(self.datanum)]
#        if 'Idx_arr' in d: # 'Idx_arr' stores the indices of valid 56*56 regions
#            print "Variable 'Idx_arr' exists in the data file!"
#            self.datanum=d['Idx_arr'].shape[1]
#            self.idx_arr=[d['Idx_arr'][0,i] for i in range(self.datanum)]
#            self.idx_flag=1
        else:
            self.datanum=self.imgnum*self.pertnum
            self.idx_arr=range(self.datanum)
#            self.idx_flag=0

        self.indim=self.param['inpsize']**2 #np.prod(np.shape(self.X[0,0]))
        self.outdim=self.param['outsize']**2 #np.prod(np.shape(self.Z[0,0]))
        self.input=[self.X[0, i] for i in range(self.imgnum)]
        self.output=[self.Z[0, i] for i in range(self.imgnum)]
        if self.param['train']=='0':
            self.output=[_*0 for _ in self.output]
        print '%d images, %d samples found' % (self.imgnum, self.datanum)

    def get_idx_arr(self):
        return self.idx_arr

    def var_check(self, thres):
        idx_arr = np.ones(self.imgnum*self.pertnum) # NOT self.datanum!
        idx_list = range( self.imgnum*self.pertnum)
        print("There are totally %d candidates" % (self.imgnum*self.pertnum)) 
        for idx in range( self.imgnum*self.pertnum ):
            img_id=idx/self.pertnum
            pert_id=idx%self.pertnum
            rot_id=pert_id%self.param['rotate']
            off_id=pert_id/self.param['rotate']
            [h, w]=self.input[img_id].shape
            [dy, dx]=self.get_offset(h, w, off_id)
            dy2=self.param['mrgsize']+dy
            dx2=self.param['mrgsize']+dx
            res2=self.output[img_id][dy2:dy2+self.param['outsize'], dx2:dx2+self.param['outsize']]
            if np.var(res2) <= thres:
                idx_list[idx] = -1
#            if idx < 10:
#                dict0 = {'Patch': res2}
#                scipy.io.savemat('./img/{}_out2.mat'.format(idx), dict0)
#            scipy.misc.imsave('./img/{}_out2.png'.format(idx), res2.astype(np.uint8))
        idx_list2 = [value for value in idx_list if value != -1]
        print("There are %d valid candidates" % len(idx_list2))
        dict1 = {'Idx_list': idx_list2}
        dict1['imgNum'] = self.imgnum
        dict1['pertNum'] = self.pertnum
        dict1['smplSide'] = self.param['smplside']
        dict1['rotate'] = self.param['rotate']
        scipy.io.savemat('Idx_list_var'+str(thres)+'_s'+str(self.param['smplside'])+'_r'+str(self.param['rotate'])+'.mat', dict1)
        print "Valid indices are stored successfully!"
        
    def get_num_images(self):
        return self.datanum
    
    def get_input_dim(self):
        return self.indim

    def get_output_dim(self):
        return self.outdim
    '''
    def get_input(self, idx):
        while 1:
            img_id=idx/self.pertnum
            pert_id=idx%self.pertnum
            rot_id=pert_id%self.param['rotate']
            off_id=pert_id/self.param['rotate']
            [h, w]=self.input[img_id].shape
            [dy, dx]=self.get_offset(h, w, off_id)
            dy2=self.param['mrgsize']+dy
            dx2=self.param['mrgsize']+dx
            res2=self.output[img_id][dy2:dy2+self.param['outsize'], dx2:dx2+self.param['outsize']]
#            if self.param['train']=='1' or np.var(res2) > 10 or idx==self.datanum-1:
            if self.param['train']=='1' or np.var(res2) > 10 or idx==self.datanum-1 or idx < 708800:
                res=self.input[img_id][dy:dy+self.param['inpsize'], dx:dx+self.param['inpsize']]
#                print '%d, %d.' % (dy, dy+self.param['inpsize'])
#                print '%d, %d.' % (dx, dx+self.param['inpsize'])
                break
            print 'Bug! idx = %d.' % idx
            idx += 1
        if rot_id==1:
            res=np.fliplr(res)
        elif rot_id==2:
            res=np.flipud(res)
        elif rot_id==3:
            res=res.T
        return res

    def get_output(self, idx):
        while 1:
            img_id=idx/self.pertnum
            pert_id=idx%self.pertnum
            rot_id=pert_id%self.param['rotate']
            off_id=pert_id/self.param['rotate']
            [h, w]=self.output[img_id].shape
            [dy, dx]=self.get_offset(h, w, off_id)
            dy+=self.param['mrgsize']
            dx+=self.param['mrgsize']
            res=self.output[img_id][dy:dy+self.param['outsize'], dx:dx+self.param['outsize']]
#            if self.param['train']=='1' or np.var(res) > 10 or idx==self.datanum-1:
            if self.param['train']=='1' or np.var(res) > 10 or idx==self.datanum-1 or idx < 708800:
                break
            print 'Bug in get_output! idx = %d.' % idx
            idx += 1
        if rot_id==1:
            res=np.fliplr(res)
        elif rot_id==2:
            res=np.flipud(res)
        elif rot_id==3:
            res=res.T
        return res
    ''' 
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
#    tmp = ts.get_idx_arr()
    ts.var_check( 100) # run variance check & store valid indices into MAT file
    '''
#    for i in range(728,748,1):
    for i in range(10):
        im=ts.get_input(i)
        y=ts.get_output(i)
        print "i={}, input={},\toutput={}".format(i, im.shape, y.shape)
#        scipy.misc.imsave('./img/{}_in.png'.format(i), im);
        scipy.misc.imsave('./img/{}_out.png'.format(i), y);
    print 'image shape:', np.shape(im)
    '''
if __name__ == '__main__':
    print 'testing ListaPrvd.py!'
    assert(len(sys.argv)==2)
    test(sys.argv[1])


