# Designed for VideoRegressionDataProvider
# Collect input data from video files!
# Filter stride = 3
# Sample size = 53
# No overlap between different samples
# 128 samples in one batch
# 63 (7 x 9) samples in one frame
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
        print "Here is in ListaPrvd_stride3.py"        

        self.indim = 57*57
        self.outdim = 45*45*16 # dummy variable! 
        self.datanum=128 # data number MAX:112, 113 failed!         
        self.frame_cnt = 9200 #700, 950, 1050
        self.sampleSize = 57
        self.sampleStride = 53 # 45 or 53

        self.cap = cv2.VideoCapture('F:/Data/AmosTV_10min_HT.divx')
        # self.cap = cv2.VideoCapture('F:/Data/Fashion_DivX720p_ASP.divx')
        self.cap.set( 1, self.frame_cnt) # Set the frame to be captured next

    def get_num_images(self):
        return self.datanum
    
    def get_input_dim(self):
        return self.indim

    def get_output_dim(self):
        return self.outdim

    def get_input(self):
        Cnt = 0
        self.input = []
        self.ref = [] # YCrCb images by bicubic interpolation       

        while(self.cap.isOpened()):  
            ret, frame = self.cap.read()
            self.frame_cnt += 2
            self.cap.set( 1, self.frame_cnt) # Set the frame to be captured next

            if not ret: #end of video
                break
            crop_frame = frame[9:9+self.sampleStride*6+self.sampleSize, 59:59+self.sampleStride*8+self.sampleSize] #[375, 481]
            down_frame = cv2.pyrDown(crop_frame, ((self.sampleStride*8+self.sampleSize+1)/2, (self.sampleStride*6+self.sampleSize+1)/2)) # downsampling captured frames [188, 241]
            down_frame_ycrcb = cv2.cvtColor(down_frame, cv2.COLOR_BGR2YCR_CB)
                    
            self.ref.append( [crop_frame, down_frame]) # when downsampling and original frames are needed
            down_frame_y = down_frame_ycrcb[:,:,0]            
            upsc_frame_y = cv2.resize(down_frame_y, (self.sampleStride*8+self.sampleSize, self.sampleStride*6+self.sampleSize), interpolation=cv2.INTER_CUBIC)
            
            for i in range(0, 7):
                for j in range(0, 9):
                    self.input.append(upsc_frame_y[self.sampleStride*i:self.sampleStride*i+self.sampleSize, self.sampleStride*j:self.sampleStride*j+self.sampleSize].astype(np.float32))
            self.input.append( np.zeros( [self.sampleSize, self.sampleSize], dtype=np.float32))
       
            Cnt += 1
            if Cnt > 1:
                break
     
        return self.input, self.ref
        
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


