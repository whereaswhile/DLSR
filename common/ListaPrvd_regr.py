# LISTA algorithm regression data provider
# obtain samples by online sampling from images
# Before using it, make sure line 167 & 170 in imgdata.py are uncommented!
# designed for multiple SR regressors

import os
import sys
import numpy as np
import scipy.misc
import scipy.io
import scipy.signal
import cPickle as pickle
import glob
import scipy.cluster.vq
from skimage.feature import hog

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
        self.param['rotate']=max(1, min(8, int(self.param['rotate'])))

        MAT_IN_VAR=self.param['MAT_IN_VAR']
        MAT_OUT_VAR=self.param['MAT_OUT_VAR']

        basename = os.path.basename(__file__)
        print os.path.splitext(basename)[0], " with paramfile:", paramfile, " (", MAT_IN_VAR, MAT_OUT_VAR, ")"

        d=scipy.io.loadmat(self.param['imgdata'])
        self.X=d[MAT_IN_VAR]
        self.Z=d[MAT_OUT_VAR]
        assert(len(np.shape(self.X))==2)
        self.imgnum=self.X.shape[1]
        self.pertnum=(self.param['smplside']**2)*self.param['rotate']

        if 'labelnum' in self.param:
            print "'labelnum' is specified!"
            labData = scipy.io.loadmat(self.param['labeldata'])
#            self.datanum = d2['Idx_list'].shape[1]
#            self.idx_arr = [d2['Idx_list'][0,i] for i in range(self.datanum)]
            self.datanum=labData[ 'label'+ self.param['labelnum'] +'_idx'].shape[1]
            print "Data Num for this label: ", self.datanum
            self.idx_arr=[ labData[ 'label'+ self.param['labelnum'] +'_idx'][0,i] for i in range(self.datanum)]
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
        scipy.io.savemat('Idx_list_var'+str(thres)+'.mat', dict1)
        print "Valid indices are stored successfully!"
        
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
            res=np.flipud(res).T
        elif rot_id==3:
            res=res.T
        elif rot_id==4:
            res=np.fliplr(res).T
        elif rot_id==5:
            res=np.flipud(res)
        elif rot_id==6:
            res=np.rot90(res,2)
        elif rot_id==7:
            res=np.rot90(res,2).T

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
            res=np.flipud(res).T
        elif rot_id==3:
            res=res.T
        elif rot_id==4:
            res=np.fliplr(res).T
        elif rot_id==5:
            res=np.flipud(res)
        elif rot_id==6:
            res=np.rot90(res,2)
        elif rot_id==7:
            res=np.rot90(res,2).T

        return res

    def getmeta(self, idx):
        return self.param
                
    def get_offset(self, h, w, i):
        iy=i/self.param['smplside']
        ix=i%self.param['smplside']
#        dy=iy*((h-self.param['inpsize'])/(self.param['smplside']-1))
#        dx=ix*((w-self.param['inpsize'])/(self.param['smplside']-1))
        dy=round(iy*float(h-self.param['inpsize'])/(self.param['smplside']-1))
        dx=round(ix*float(w-self.param['inpsize'])/(self.param['smplside']-1))
        return dy, dx

def getStore(param):
    return ListaSet(param)


def varThreshold(param, thres):
    ts = ListaSet(param)
    print "{} images in total".format(ts.get_num_images())

#    ts.var_check( 100) # run variance check & store valid indices into MAT file
    idx_list = range( ts.get_num_images())
    for idx in range( ts.get_num_images()):
        patch = ts.get_input( idx)

        if np.var( patch) <= thres:
            idx_list[idx] = -1

    idx_list2 = [value for value in idx_list if value != -1]
    print("There are %d valid candidates" % len(idx_list2))
    idx_arr = np.asarray( idx_list2)
    np.save( 'regressor2/test_idx', idx_arr)    

def transferData(param):
    ts = ListaSet(param)
    print "{} images in total".format(ts.get_num_images())

    d1 = {}
    d1['Input'] = np.empty([13, 13, ts.get_num_images()], dtype=np.float32)
    d1['Output'] = np.empty([1, 1, ts.get_num_images()], dtype=np.float32)
    for i in range( ts.get_num_images()):
        d1['Input'][:,:,i] = ts.get_input(i)
        d1['Output'][:,:,i] = ts.get_output(i)

#    scipy.io.savemat( 'trainingPatches13x13.mat', d1)
    scipy.io.savemat( 'testPatches13x13.mat', d1)
    print "Data stored successfully!"

def hogKmeansClustering(param, orientNum, DATA_FOLDER):
    ts = ListaSet(param)
    print "{} images in total".format(ts.get_num_images())

    clustNum = 5
#    d1 = {}
#    Patches = np.empty([ts.get_num_images(), 13, 13 ], dtype=np.float32)
    PatchArr = np.empty([ ts.get_num_images(), orientNum])
    for i in range( ts.get_num_images()):
#        Patches[i,:,:] = ts.get_input(i)
        PatchArr[i, :] = hog( ts.get_input(i), orientations=orientNum, pixels_per_cell=(13, 13), cells_per_block=(1, 1), visualise=False)
        # for 5x5 centered at 13x13 input    
#        PatchArr[i, :] = hog( ts.get_input(i)[4:9, 4:9], orientations=orientNum, pixels_per_cell=(5, 5), cells_per_block=(1, 1), visualise=False)
        # for 17x17 input
#        PatchArr[i, :] = hog( ts.get_input(i)[6:11, 6:11], orientations=orientNum, pixels_per_cell=(5, 5), cells_per_block=(1, 1), visualise=False)
        if i % 10000 == 0:
            print i

    print "load existing codebook..."
    codebook = np.load( DATA_FOLDER+'/4bins_5cls/codebookArtificial2.npy')
    print codebook

    print "assign codes for patches..."
    code, dist = scipy.cluster.vq.vq( PatchArr, codebook)

    d1 = {}
    for i in range( clustNum):
        d1['label'+str(i+1)+'_idx'] = np.where( code == i)[0].astype( np.uint32)
    d1['labelArr'] = code.astype( np.uint32)
#        d1['PatchArr'+str(i+1)] = Patches[ np.where( code == i)[0], :, :].astype( np.uint32)
#    d1['PatchArr'] = Patches

    scipy.io.savemat( DATA_FOLDER+'/4bins_5cls/trainlabelsArtificial13code2_sps100_r8.mat', d1)
#    scipy.io.savemat( DATA_FOLDER+'/4bins_5cls/testlabelsArtificial13c5code3.mat', d1)
#    scipy.io.savemat( DATA_FOLDER+'/4bins_5cls/testPatches.mat', d1)
#    scipy.io.savemat( DATA_FOLDER+'/4bins_5cls/testpatches_sps200.mat', d1)

# clustering according to specific filters
def Cluster(param, DATA_FOLDER):
    ts = ListaSet(param)
    Data = scipy.io.loadmat('./data/filter_template3.mat')
    Filter2 = np.rot90( Data['Filter2'], 2)

#    corList = []
#    TVList= []
    corArr = np.empty(ts.get_num_images())
    TVArr = np.empty(ts.get_num_images())

    for i in range( ts.get_num_images()):
#    for i in range( 100):
        tmp = ts.get_input(i)
        tmp2 = tmp - np.mean( tmp)
        tmp3 = tmp2 / np.linalg.norm(tmp2, 'fro')
#        Cor = scipy.signal.convolve2d(tmp3, Filter2, 'valid')
#        corList.append( Cor)    
        corArr[i] = scipy.signal.convolve2d(tmp3, Filter2, 'valid')
    
        dx = scipy.ndimage.sobel(tmp, 0)
        dy = scipy.ndimage.sobel(tmp, 1)
        mag = np.hypot(dx, dy)
#        TVList.append( np.sum(mag))
        TVArr[i] = np.sum(mag)

        if i % 10000 == 0:
            print i

    np.save( DATA_FOLDER+'/trainCorrelation.npy', corArr)
    np.save( DATA_FOLDER+'/trainTotalVariation.npy', TVArr)
    return

def getClusterLabel(param, trainFlag, DATA_FOLDER):
    if trainFlag == '1':
        corArr = np.load( DATA_FOLDER+'/trainCorrelation.npy') 
        TVArr = np.load( DATA_FOLDER+'/trainTotalVariation.npy') 
    else:
        corArr = np.load( DATA_FOLDER+'/testCorrelation.npy') 
        TVArr = np.load( DATA_FOLDER+'/testTotalVariation.npy') 
    
    corThres = 0.85
    TVThres = 0

#    print (corArr >= corThres)
#    PatchIdx = np.where( [ x&y for (x, y) in zip(corArr >= corThres, TVArr >= TVThres)] )[0].astype( np.uint32)
#    print PatchIdx.shape
#    print PatchIdx[:10]
    d1 = {}
    i = 0
#    d1['label'+str(i+1)+'_idx'] = np.where( code == i)[0].astype( np.uint32)
    d1['label'+str(i+1)+'_idx'] = np.where( [ x&y for (x, y) in zip(corArr >= corThres, TVArr >= TVThres)] )[0].astype( np.uint32)
    if trainFlag == '1':
        scipy.io.savemat( DATA_FOLDER+'/trainlabels_TVThres0_sps150.mat', d1)
    else:
        scipy.io.savemat( DATA_FOLDER+'/testlabels_TVThres0_sps200.mat', d1)
    
# k-means clustering of training samples
def kmeans(param):
    ts = ListaSet(param)
    print "{} images in total".format(ts.get_num_images())

    f1 = np.array([[-1,0,1]])
    f3 = np.array([[1,0,-2,0,1]])
    patchList = []
    for i in range(ts.get_num_images()):
        patch = ts.get_input(i)       
#        print patch
#        exit() 
        lfea = np.zeros((4, patch.shape[0]-4, patch.shape[1]-4))
        lfea[0,:,:] = scipy.signal.convolve2d( patch, f1, mode='same')[2:-2, 2:-2]
        lfea[1,:,:] = scipy.signal.convolve2d( patch, f1.T, mode='same')[2:-2, 2:-2]
        lfea[2,:,:] = scipy.signal.convolve2d( patch, f3, mode='same')[2:-2, 2:-2]
        lfea[3,:,:] = scipy.signal.convolve2d( patch, f3.T, mode='same')[2:-2, 2:-2]

        patchsList = []
#        for j in range( lfea.shape[1]-5+1):
#            for k in range( lfea.shape[2]-5+1):
##                print (j, k)
#                sfea = lfea[:,j:j+5, k:k+5]
##                tmp = sfea.flatten() / np.linalg.norm( sfea) 
##                print np.linalg.norm(tmp)
##                exit()
        sfea = lfea[:,2:7, 2:7]
        patchsList.append( sfea.flatten() / (np.linalg.norm( sfea)+1e-15))

        lfea2 = np.hstack( s for s in patchsList)
#        print lfea2.shape
#        exit()
        patchList.append( lfea2)    
    patchArr = np.vstack( sl for sl in patchList)
    print "patch feature array dimension: ", patchArr.shape
    del patchList # to save memory!

    # k-means clustering
#    whitened = scipy.cluster.vq.whiten( patchArr)
    clustNum = 4
    print "finding the codebook..."
    codebook, err = scipy.cluster.vq.kmeans( patchArr, clustNum, iter=2)
#    print codebook.shape
    print "assign codes for patches..."
    code, dist = scipy.cluster.vq.vq( patchArr, codebook)
#    print code.shape
#    for i in range(20):
#        print code[i]
    
    d1 = {}
    for i in range(4):
        d1['label'+str(i+1)+'_idx'] = np.where( code == i)[0].astype( np.uint32)

    scipy.io.savemat( 'regressor2/trainlabels.mat', d1)

def drawSamples(param):
    ts = ListaSet(param)
    print "{} images in total".format(ts.get_num_images())

    idxList = [3]
#    idxList = [138016, 464931, 432135, 525338, 698827, 188658, 131693, 620405]
    patchList = []
    for i in idxList:
        patchList.append( ts.get_input( i))
    d1 = {'Input': patchList}
#    scipy.io.savemat( 'TrainPatch.mat', d1)
    scipy.io.savemat( 'TestPatch.mat', d1)
    return

    d = scipy.io.loadmat( 'labelsMinError.mat')
    d1 = {}
    for j in range(4):
        labelList = [ d['label'+str(j+1)+'Arr'][0,i] for i in range(d['label'+str(j+1)+'Arr'].shape[1])]
        patchList = []
        for i in range( len(labelList)):
            patchList.append( ts.get_input( labelList[i]))
        d1['patchList'+str(j+1)] = patchList
#    scipy.io.savemat( 'patchesMinError.mat', d1)

def drawAnchorSamples(param, thres):
    ts = ListaSet(param)
    print "{} images in total".format(ts.get_num_images())
    TrainLabelArr = np.load( 'regressor2/anchor_label2k_5.npy')
    TrainLabelList = TrainLabelArr.tolist()
    TrainLabelList = range( ts.get_num_images())
#    print TrainLabelList
    print len(TrainLabelList)
#    NewList = list( TrainLabelList)
    for cnt in range( len( TrainLabelList)):
#        print cnt
        patch = ts.get_input( TrainLabelList[cnt])
    
        if np.var( patch) <= thres:
            TrainLabelList[cnt] = -1
    
    LabelList2 = [value for value in TrainLabelList if value != -1]    
    print len( LabelList2)
    LabelArr2 = np.asarray( LabelList2)
#    print LabelArr2
    np.save('regressor2/anchor_label_var.npy', LabelArr2)
    return
  
def test(param):
    ts = ListaSet(param)
    print "{} images in total".format(ts.get_num_images())
    patchList = []
#    for i in range(100000):
    for i in range( ts.get_num_images()):
        tmp = ts.get_input( i)
        tmp = tmp.flatten() - tmp.mean()
        patchList.append( tmp)
    patchArr = np.vstack( patch for patch in patchList)
    print patchArr.shape
#    np.save( 'TestPatchArr', patchArr)
    np.save( 'TrainingPatchArr', patchArr)

if __name__ == '__main__':
    print 'testing ListaPrvd.py!'
    assert(len(sys.argv) >= 2)
#    varThreshold(sys.argv[1], 100)
#    kmeans(sys.argv[1])
#    drawSamples(sys.argv[1])
#    test(sys.argv[1])
#    drawAnchorSamples(sys.argv[1], 100)

#    transferData( sys.argv[1])
    hogKmeansClustering( sys.argv[1], 4, './hogCluster' )

#    Cluster( sys.argv[1], './filterCluster')
#    getClusterLabel( sys.argv[1], sys.argv[2], './filterCluster')
