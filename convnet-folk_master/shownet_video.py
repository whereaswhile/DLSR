# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The class "ShowConvNet_video" is designed for processing general video files.

import numpy
import sys
import getopt as opt
from util import *
from math import sqrt, ceil, floor
import os
from gpumodel import IGPUModel
import random as r
import numpy.random as nr
from convnet import ConvNet
from options import *
from data import DataProvider, dp_types

from shownet import ShowConvNet, ShowNetError
import scipy.io as sio
import cv2
# import imp
from w_util import readLines

try:
    import pylab as pl
except:
    print "This script requires the matplotlib python library (Ubuntu/Fedora package name python-matplotlib). Please install it."
#    sys.exit(1)

import matplotlib.cm as cm


class ShowConvNet_video( ShowConvNet):
    def __init__(self, op, load_dic):
        ShowConvNet.__init__(self, op, load_dic)
    
    def do_write_features(self):
        if len(self.feature_path)==0: #evaluate only
            print "evaluation mode, no feature will be saved"
            nacc = 0
            ncnt = 0
            self.mult_view=max(1, self.mult_view)
        elif not os.path.exists(self.feature_path):
            os.makedirs(self.feature_path)
        next_data = self.get_next_batch(train=False)
        b1 = next_data[1]
        num_ftrs = self.layers[self.ftr_layer_idx]['outputs']
        data_dims = [_.shape[0] for _ in next_data[2]]
        print "input data dimensions: {}".format(data_dims)
        if data_dims.count(1)==1:
            label_idx = data_dims.index(1)
        elif data_dims.count(1)==0: # regression data
            label_idx = 1
        else:
            raise ShowNetError("data_dims invalid format!")
            print "writing features: layer idx={}, {} fitlers, label_idx={}".format(self.ftr_layer_idx, num_ftrs, label_idx)
            print "starting from batch: {}".format(b1)
            
        # Newly added
        sampleSize = 56
        sampleStride = 44 # response: output of ConvNet
        # Parse input paramters from file
        self.data_param = {'paramfile': self.data_path_test}
        plines=readLines(self.data_path_test)
        for l in plines:
            l = l.rstrip().split()
            self.data_param[l[0]] = l[1]
        
        self.cap = cv2.VideoCapture(self.data_param['videoName'])    
        self.cap.set( 1, int(self.data_param['startingFrmNo'])) # Set the frame to be captured next

        frmWidth = int( self.cap.get(3))
        frmHeight = int( self.cap.get(4))
        frmRate = int( self.cap.get(5))

        heightNum = int( numpy.ceil( float(frmHeight*2)/ sampleStride))
        widthNum = int( numpy.ceil( float(frmWidth*2)/ sampleStride))

        h_mark = numpy.rint( numpy.linspace(0, frmHeight*2-sampleSize, num=heightNum)) # height mark
        w_mark = numpy.rint( numpy.linspace(0, frmWidth*2-sampleSize, num=widthNum)) # width mark
        h_cnt = 0; # height count
        w_cnt = 0; # width count
        
        video = cv2.VideoWriter(self.data_param['saveName'], -1, frmRate, (frmWidth*2, frmHeight*2)) # size designed for uncompressed
        if video.isOpened():
            print 'Video is initialized successfully!'
        else:
            print 'Video is NOT initialized!'
            sys.exit(1)
            
        frame_y = numpy.zeros( (frmHeight*2-12, frmWidth*2-12), dtype=numpy.float32)
        cntMap = numpy.zeros( (frmHeight*2-12, frmWidth*2-12))
        cntMapMark = 0 # 0: can be accumulated, 1: cannot be accumulated
        while True:
            batch = next_data[1]
            data = next_data[2]
            ftrs = n.zeros((data[0].shape[1], num_ftrs), dtype=n.single)
            self.libmodel.startFeatureWriter(data + [ftrs], self.ftr_layer_idx)

            # load the next batch while the current one is computing
            next_data = self.get_next_batch(train=False)
            self.finish_batch()
            
            for i in range( ftrs.shape[0]):
                h_idx = h_mark[h_cnt]
                w_idx = w_mark[w_cnt]

                frame_y[h_idx:h_idx+sampleStride, w_idx:w_idx+sampleStride] += \
                    n.swapaxes(n.reshape( ftrs[i, 0:sampleStride*sampleStride], [sampleStride,sampleStride], order='F'), 0,1)
                if cntMapMark == 0:
                    cntMap[h_idx:h_idx+sampleStride, w_idx:w_idx+sampleStride] += n.ones( (sampleStride,sampleStride))
                # increase indices
                if w_cnt < len(w_mark)-1:
                    w_cnt += 1
                elif h_cnt < len(h_mark)-1:
                    h_cnt += 1
                    w_cnt = 0
                else:
                    # store this frame_y
                    frame_y_float = frame_y/cntMap 
                    frame_y = float2uint8( frame_y_float)
                    ret, currentFrm = self.cap.read()
                    if not ret: #end of video
                        print 'Fail to read next frame in video. This cannot happen!'
                        sys.exit(1)

                    frame_ycrcb = cv2.cvtColor(currentFrm, cv2.COLOR_BGR2YCR_CB)
                    frame_bicubic_ycrcb = cv2.resize(frame_ycrcb, (int(frmWidth)*2, int(frmHeight)*2), interpolation=cv2.INTER_CUBIC)
                    frame_bicubic_ycrcb[6:frmHeight*2-6,6:frmWidth*2-6,0] = frame_y
                    frame_bicubic = cv2.cvtColor(frame_bicubic_ycrcb, cv2.COLOR_YCR_CB2BGR)
                    video.write( frame_bicubic)
                    # cv2.imwrite('frame.png', frame_bicubic) # print out for debug
                    # print 'Write a frame!'
                    h_cnt = 0
                    w_cnt = 0
                    cntMapMark = 1 # keep 'cntMap' unchanged afterwards
                    frame_y = numpy.zeros( (frmHeight*2-12, frmWidth*2-12), dtype=numpy.float32) # reset 'frame_y'
                    
            # output = {'source_model':self.load_file, 'num_vis':num_ftrs, 'data': ftrs, 'labels': data[label_idx]}
            # try:
                # output['aux'] = self.test_data_provider.getftraux()
            # except AttributeError:
                # pass

            # if len(self.feature_path)==0: #evaluate only
                # nacc, ncnt=self.increase_acc_count(ftrs, data[label_idx][0], nacc, ncnt)
                # print "Batch %d evaluated: %.2f" % (batch, 1.0*nacc/ncnt*100)
            # else:
                # path_out = os.path.join(self.feature_path, 'data_batch_%d' % batch)
                # pickle(path_out,output)
                # print "Wrote feature file %s" % path_out
            sys.stdout.flush()

            if next_data[1] == b1:
                break
 
        # if len(self.feature_path)==0: #evaluate only
            # print "overall accuracy: %.3f%%" % (1.0*nacc/ncnt*100)

        video.release()
 
def float2uint8( tmp1): # convert float32 to uint8 in Matlab fashion
    l_idx = tmp1 < 0
    tmp1[l_idx] = 0
    h_idx = tmp1 > 255
    tmp1[h_idx] = 255
    tmp2 = numpy.uint8(numpy.rint(tmp1))
    return tmp2
 
if __name__ == "__main__":
    try:
        op = ShowConvNet.get_options_parser()
        op, load_dic = IGPUModel.parse_options(op)
        # model = ShowConvNet(op, load_dic)
        model = ShowConvNet_video(op, load_dic)
        model.start()
    except (UnpickleError, ShowNetError, opt.GetoptError), e:
        print "----------------"
        print "Error:"
        print e 
