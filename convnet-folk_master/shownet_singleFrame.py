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

# The class "ShowConvNet_singleFrame" is specially designed for processing one single image super-resolution.
# It loads .mat file as input and save the final response/output of CNN in .mat file

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


class ShowConvNet_singleFrame( ShowConvNet):
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
        respSize = 44 # response: output of ConvNet
        self.data_param = {'paramfile': self.data_path_test}
        plines=readLines(self.data_path_test)
        for l in plines:
            l = l.rstrip().split()
            self.data_param[l[0]] = l[1]
        print self.data_param
        d2 = sio.loadmat(self.data_param['imgdata'])
        
        Data = numpy.empty( [0,respSize*respSize])        
        
        while True:
            batch = next_data[1]
            data = next_data[2]
            ftrs = n.zeros((data[0].shape[1], num_ftrs), dtype=n.single)
            self.libmodel.startFeatureWriter(data + [ftrs], self.ftr_layer_idx)
            
            # load the next batch while the current one is computing
            next_data = self.get_next_batch(train=False)
            self.finish_batch()
            output = {'source_model':self.load_file, 'num_vis':num_ftrs, 'data': ftrs, 'labels': data[label_idx]}
            
            Data0 = ftrs[:, 0:respSize*respSize]
            Data = numpy.concatenate( (Data, Data0), axis=0)

            # try:
                # output['aux'] = self.test_data_provider.getftraux()
            # except AttributeError:
                # pass
            
            if len(self.feature_path)==0: #evaluate only
                nacc, ncnt=self.increase_acc_count(ftrs, data[label_idx][0], nacc, ncnt)
                print "Batch %d evaluated: %.2f" % (batch, 1.0*nacc/ncnt*100)
            else:
                path_out = os.path.join(self.feature_path, 'data_batch_%d' % batch)
                # pickle(path_out,output)
                print "Wrote feature file %s" % path_out
            sys.stdout.flush()

            if next_data[1] == b1:
                break
        
        height_idx = d2['Height_idx']
        width_idx = d2['Width_idx']
        
        img = numpy.zeros( (d2['Height']-12, d2['Width']-12), dtype=numpy.float32)
        cntMap = numpy.zeros( (d2['Height']-12, d2['Width']-12))

        for i in range( d2['HeightNum']):
            for j in range( d2['WidthNum']):
                img[height_idx[0,i]-1:height_idx[0,i]-1+respSize, width_idx[0,j]-1:width_idx[0,j]-1+respSize] \
                    += numpy.swapaxes( numpy.reshape( Data[i*int( d2['WidthNum'])+j, :], [respSize,respSize], order='F'), 0,1)
                cntMap[height_idx[0,i]-1:height_idx[0,i]-1+respSize, width_idx[0,j]-1:width_idx[0,j]-1+respSize] += numpy.ones( (respSize,respSize))

        img2 = img/cntMap
        img2 = float2uint8( img2)
        # cv2.imwrite( 'results.png', img2) # print out for debug

        dict1 = {}
        dict1['Img_y'] = img2
        # dict1['Patch'] = Data
        
        if 'saveName' in self.data_param:
            sio.savemat( self.data_param['saveName'], dict1)
            print "Wrote response .mat file %s" % self.data_param['saveName']
        else:
            print "Saving name of response .mat file is missing!"
        
        if len(self.feature_path)==0: #evaluate only
            print "overall accuracy: %.3f%%" % (1.0*nacc/ncnt*100)

 
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
        model = ShowConvNet_singleFrame(op, load_dic)
        model.start()
    except (UnpickleError, ShowNetError, opt.GetoptError), e:
        print "----------------"
        print "Error:"
        print e 
