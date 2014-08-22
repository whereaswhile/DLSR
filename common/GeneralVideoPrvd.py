# Designed for VideoRegressionDataProvider
# Collect input data from general video files!

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
        # print self.param    
        self.cap = cv2.VideoCapture(self.param['videoName'])    
        self.cap.set( 1, int( self.param['startingFrmNo'])) # Set the frame to be captured next
        
        self.frmWidth = int( self.cap.get(3))
        self.frmHeight = int( self.cap.get(4))

        self.sampleSize = 56
        self.sampleStride = 44 # 45 or 53
        self.HeightNum = int( np.ceil( float(self.frmHeight*2)/ self.sampleStride))
        self.WidthNum = int( np.ceil( float(self.frmWidth*2)/ self.sampleStride))
        self.h_mark = np.rint( np.linspace(0, self.frmHeight*2-self.sampleSize, num=self.HeightNum)) # height mark
        self.w_mark = np.rint( np.linspace(0, self.frmWidth*2-self.sampleSize, num=self.WidthNum)) # width mark

        self.datanum = ( int(self.param['endingFrmNo'])-int(self.param['startingFrmNo'])+1)*self.HeightNum*self.WidthNum        
        self.indim = self.sampleSize*self.sampleSize
        self.outdim = self.sampleStride*self.sampleStride # dummy variable! 

        self.h_cnt = 0; # height count
        self.w_cnt = 0; # width count

    def read_next_frame(self):
        ret, self.currentFrm = self.cap.read()
        if not ret: #end of video
            print 'Fail to read next frame in video. This cannot happen!'
            sys.exit(1)

        frame_ycrcb = cv2.cvtColor(self.currentFrm, cv2.COLOR_BGR2YCR_CB)
        frame_bicubic_ycrcb = cv2.resize(frame_ycrcb, (int(self.frmWidth)*2, int(self.frmHeight)*2), interpolation=cv2.INTER_CUBIC)
        self.frame_bicubic_y = frame_bicubic_ycrcb[:,:,0]
        
    def get_num_images(self):
        return self.datanum
    
    def get_input_dim(self):
        return self.indim

    def get_output_dim(self):
        return self.outdim

    def get_input(self):        
        if self.h_cnt == 0 and self.w_cnt == 0:
            self.read_next_frame()
        
        h_idx = self.h_mark[self.h_cnt]
        w_idx = self.w_mark[self.w_cnt]
        
        # increase indices
        if self.w_cnt < len(self.w_mark)-1:
            self.w_cnt += 1
        elif self.h_cnt < len(self.h_mark)-1:
            self.h_cnt += 1
            self.w_cnt = 0
        else:
            self.h_cnt = 0
            self.w_cnt = 0
        
        return self.frame_bicubic_y[h_idx:h_idx+self.sampleSize, w_idx:w_idx+self.sampleSize]    
        
    def release_cam(self):
        self.cap.release()
        print 'Successfully release the camera!'

    # def get_output(self):
        # self.output = []

        # return self.output[idx]
        # return self.output[0]
        # return np.zeros((44,44,16), dtype = np.single)
    
    # def getmeta(self, idx):
        # return self.param
                
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


