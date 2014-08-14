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

# Modified for showing SR artifacts of videos
# Note: Sewing patches into one single image is implemented in Python! 
# Designed for stride = 3
# (1) All overlap (is commentted)
# (2) no overlap between 64 samples within 1 frame

# import numpy
import sys
import getopt as opt
from util import *
from math import sqrt, ceil, floor, log10
import os
from gpumodel import IGPUModel
import random as r
import numpy.random as nr
from convnet import ConvNet
from options import *
from data import DataProvider, dp_types

import time
from numpy import *
import cv2
import scipy.io
import threading

try:
    import pylab as pl
except:
    print "This script requires the matplotlib python library (Ubuntu/Fedora package name python-matplotlib). Please install it."
#    sys.exit(1)

import matplotlib.cm as cm

class ShowNetError(Exception):
    pass

class ShowConvNet(ConvNet):
    def __init__(self, op, load_dic):
        ConvNet.__init__(self, op, load_dic)
    
    def get_gpus(self):
        self.need_gpu = self.op.get_value('show_preds') or self.op.get_value('write_features') or self.op.get_value('write_pixel_proj')
        if self.need_gpu:
            ConvNet.get_gpus(self)
    
    def init_data_providers(self):
        class Dummy:
            def advance_batch(self):
                pass
        if self.need_gpu:
            ConvNet.init_data_providers(self)
        else:
            self.train_data_provider = self.test_data_provider = Dummy()
    
    def import_model(self):
        if self.need_gpu:
            ConvNet.import_model(self)

    def init_data_providers(self):
        self.dp_params['convnet'] = self
        self.dp_params['imgprovider'] = self.img_provider_file
        try:
            if self.need_gpu:
                self.test_data_provider = DataProvider.get_instance(self.data_path_test, self.test_batch_range,
                                                                    type=self.dp_type_test, dp_params=self.dp_params, test=True)
                
                self.test_batch_range = self.test_data_provider.batch_range
        except Exception, e:
            print "Unable to create data provider: %s" % e
            self.print_data_providers()
            sys.exit()
            
    def init_model_state(self):
        #ConvNet.init_model_state(self)
        if self.op.get_value('show_preds'):
            self.sotmax_idx = self.get_layer_idx(self.op.get_value('show_preds'), check_type='softmax')
        if self.op.get_value('write_features'):
            self.ftr_layer_idx = self.get_layer_idx(self.op.get_value('write_features'))
        if self.op.get_value('write_pixel_proj'):
            tmp = self.op.get_value('write_pixel_proj')
            tmp = tmp.split[',']
            self.ftr_layer_idx = self.get_layer_idx[tmp[0]]
            self.ftr_res_idx = int(tmp[1])
            
    def init_model_lib(self):
        if self.need_gpu:
            if self.op.get_value('write_pixel_proj'):
                # in pixel projection model, activation matrix cannot be shared 
                for l in self.model_state['layers']:
                    l['usesActs'] = True
                
            ConvNet.init_model_lib(self)

    def plot_cost(self):
        if self.show_cost not in self.train_outputs[0][0]:
            raise ShowNetError("Cost function with name '%s' not defined by given convnet." % self.show_cost)
        train_errors = [o[0][self.show_cost][self.cost_idx] for o in self.train_outputs]
        test_errors = [o[0][self.show_cost][self.cost_idx] for o in self.test_outputs]

        #        numbatches = len(self.train_batch_range)
        numbatches = self.model_state['batchnum']
        test_errors = numpy.row_stack(test_errors)
        test_errors = numpy.tile(test_errors, (1, self.testing_freq))
        test_errors = list(test_errors.flatten())
        test_errors += [test_errors[-1]] * max(0,len(train_errors) - len(test_errors))
        test_errors = test_errors[:len(train_errors)]

        numepochs = len(train_errors) / float(numbatches)
        print numepochs, numbatches
        pl.figure(1)
        x = range(0, len(train_errors))
        pl.plot(x, train_errors, 'k-', label='Training set')
        pl.plot(x, test_errors, 'r-', label='Test set')
        pl.legend()
        ticklocs = range(numbatches, len(train_errors) - len(train_errors) % numbatches + 1, numbatches)
        epoch_label_gran = int(ceil(numepochs / 20.)) # aim for about 20 labels
        epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10) # but round to nearest 10
        ticklabels = map(lambda x: str((x[1] / numbatches)) if x[0] % epoch_label_gran == epoch_label_gran-1 else '', enumerate(ticklocs))

        pl.xticks(ticklocs, ticklabels)
        pl.xlabel('Epoch')
#        pl.ylabel(self.show_cost)
        pl.title(self.show_cost)
        
    def make_filter_fig(self, filters, filter_start, fignum, _title, num_filters, combine_chans):
        FILTERS_PER_ROW = 16
        MAX_ROWS = 16
        MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS
        num_colors = filters.shape[0]
        f_per_row = int(ceil(FILTERS_PER_ROW / float(1 if combine_chans else num_colors)))
        filter_end = min(filter_start+MAX_FILTERS, num_filters)
        filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))
    
        filter_size = int(sqrt(filters.shape[1]))
        fig = pl.figure(fignum)
        fig.text(.5, .95, '%s %dx%d filters %d-%d' % (_title, filter_size, filter_size, filter_start, filter_end-1), horizontalalignment='center') 
        num_filters = filter_end - filter_start
        if not combine_chans:
            bigpic = n.zeros((filter_size * filter_rows + filter_rows + 1, filter_size*num_colors * f_per_row + f_per_row + 1), dtype=n.single)
        else:
            bigpic = n.zeros((3, filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=n.single)
    
        for m in xrange(filter_start,filter_end ):
            filter = filters[:,:,m]
            y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
            if not combine_chans:
                for c in xrange(num_colors):
                    filter_pic = filter[c,:].reshape((filter_size,filter_size))
                    bigpic[1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                           1 + (1 + filter_size*num_colors) * x + filter_size*c:1 + (1 + filter_size*num_colors) * x + filter_size*(c+1)] = filter_pic
            else:
                filter_pic = filter.reshape((3, filter_size,filter_size))
                bigpic[:,
                       1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                       1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
                
        pl.xticks([])
        pl.yticks([])
        if not combine_chans:
            pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
        else:
            bigpic = bigpic.swapaxes(0,2).swapaxes(0,1)
            pl.imshow(bigpic, interpolation='nearest')        
        
    def plot_filters(self):
        filter_start = 0 # First filter to show
        layer_names = [l['name'] for l in self.layers]
        if self.show_filters not in layer_names:
            raise ShowNetError("Layer with name '%s' not defined by given convnet." % self.show_filters)
        layer = self.layers[layer_names.index(self.show_filters)]
        filters = layer['weights'][self.input_idx]
        if layer['type'] == 'fc': # Fully-connected layer
            num_filters = layer['outputs']
            channels = self.channels
        elif layer['type'] in ('conv', 'local'): # Conv layer
            num_filters = layer['filters']
            channels = layer['filterChannels'][self.input_idx]
            if layer['type'] == 'local':
                filters = filters.reshape((layer['modules'], layer['filterPixels'][self.input_idx] * channels, num_filters))
                filter_start = r.randint(0, layer['modules']-1)*num_filters # pick out some random modules
                filters = filters.swapaxes(0,1).reshape(channels * layer['filterPixels'][self.input_idx], num_filters * layer['modules'])
                num_filters *= layer['modules']

        filters = filters.reshape(channels, filters.shape[0]/channels, filters.shape[1])
        # Convert YUV filters to RGB
        if self.yuv_to_rgb and channels == 3:
            R = filters[0,:,:] + 1.28033 * filters[2,:,:]
            G = filters[0,:,:] + -0.21482 * filters[1,:,:] + -0.38059 * filters[2,:,:]
            B = filters[0,:,:] + 2.12798 * filters[1,:,:]
            filters[0,:,:], filters[1,:,:], filters[2,:,:] = R, G, B
        combine_chans = not self.no_rgb and channels == 3
        
        # Make sure you don't modify the backing array itself here -- so no -= or /=
        filters = filters - filters.min()
        filters = filters / filters.max()

        self.make_filter_fig(filters, filter_start, 2, 'Layer %s' % self.show_filters, num_filters, combine_chans)
    
    def plot_predictions(self):
        data = self.get_next_batch(train=False)[2] # get a test batch
        num_classes = self.test_data_provider.get_num_classes()
        NUM_ROWS = 2
        NUM_COLS = 4
        NUM_IMGS = NUM_ROWS * NUM_COLS
        NUM_TOP_CLASSES = min(num_classes, 4) # show this many top labels
        
        label_names = self.test_data_provider.batch_meta['label_names']
        if self.only_errors:
            preds = n.zeros((data[0].shape[1], num_classes), dtype=n.single)
        else:
            preds = n.zeros((NUM_IMGS, num_classes), dtype=n.single)
            rand_idx = nr.randint(0, data[0].shape[1], NUM_IMGS)
            data[0] = n.require(data[0][:,rand_idx], requirements='C')
            data[1] = n.require(data[1][:,rand_idx], requirements='C')
        data += [preds]

        # Run the model
        self.libmodel.startFeatureWriter(data, self.sotmax_idx)
        self.finish_batch()
        
        fig = pl.figure(3)
        fig.text(.4, .95, '%s test case predictions' % ('Mistaken' if self.only_errors else 'Random'))
        if self.only_errors:
            err_idx = nr.permutation(n.where(preds.argmax(axis=1) != data[1][0,:])[0])[:NUM_IMGS] # what the net got wrong
            data[0], data[1], preds = data[0][:,err_idx], data[1][:,err_idx], preds[err_idx,:]
            
        data[0] = self.test_data_provider.get_plottable_data(data[0])
        for r in xrange(NUM_ROWS):
            for c in xrange(NUM_COLS):
                img_idx = r * NUM_COLS + c
                if data[0].shape[0] <= img_idx:
                    break
                pl.subplot(NUM_ROWS*2, NUM_COLS, r * 2 * NUM_COLS + c + 1)
                pl.xticks([])
                pl.yticks([])
                greyscale = False
                try:
                    img = data[0][img_idx,:,:,:]
                except IndexError:
                    # maybe greyscale?
                    greyscale = True
                    img = data[0][img_idx,:,:]
                if len(img.shape) == 3 and img.shape[2]==1:
                    img = img.reshape(img.shape[:2])
                    greyscale = True
                if not greyscale:
                    pl.imshow(img, interpolation='nearest')
                else:
                    pl.imshow(img, interpolation='nearest', cmap=cm.Greys_r)

                true_label = int(data[1][0,img_idx])

                img_labels = sorted(zip(preds[img_idx,:], label_names), key=lambda x: x[0])[-NUM_TOP_CLASSES:]
                pl.subplot(NUM_ROWS*2, NUM_COLS, (r * 2 + 1) * NUM_COLS + c + 1, aspect='equal')

                ylocs = n.array(range(NUM_TOP_CLASSES)) + 0.5
                height = 0.5
                width = max(ylocs)
                pl.barh(ylocs, [l[0]*width for l in img_labels], height=height, \
                        color=['r' if l[1] == label_names[true_label] else 'b' for l in img_labels])
                pl.title(label_names[true_label])
                pl.yticks(ylocs + height/2, [l[1] for l in img_labels])
                pl.xticks([width/2.0, width], ['50%', ''])
                pl.ylim(0, ylocs[-1] + height*2)
    
    def increase_acc_count(self, pred, label, nacc, ncnt):
		if ncnt==0:
			self.remain_pred=pred[0:0]
			self.remain_label=label[0:0]
		pred=numpy.concatenate((self.remain_pred, pred), axis=0)
		label=numpy.concatenate((self.remain_label, label), axis=0)
		idx=range(0, len(pred), self.mult_view)
		if len(pred)%self.mult_view!=0:
			idx=idx[:-1]
		for i in idx:
			#print 'i=', i, label[i:i+self.mult_view]
			assert len(set(label[i:i+self.mult_view].tolist()))==1
			ncnt+=1
			if numpy.argmax(numpy.sum(pred[i:i+self.mult_view], axis=0))==label[i]:
				nacc+=1;
		self.remain_pred=pred[len(idx)*self.mult_view:]
		self.remain_label=label[len(idx)*self.mult_view:]
		#print 'remain: ', self.remain_label
		return (nacc, ncnt)
    
    def get_next_batch_video(self, train=True):
        # print 'Here is get_next_batch_video in shownet_stride3.py'
        if not train:
            dp = self.test_data_provider           
        else:
            dp = self.train_data_provider      
        return self.parse_batch_data(dp.get_next_batch_video(), train=train) # dp.get_next_batch_video is in imgdata.py
        
    def close_camera(self):
        dp = self.test_data_provider
        dp.close_camera()
    
    def do_write_features(self):
        self.orinList = []
        self.lresList = []
        self.hresList = []
        self.bicuList = []
        
        d=scipy.io.loadmat('CntMat.mat') # load CountMap
       
        next_data = self.get_next_batch_video(train=False) ### get_next_batch is in this code file!
        num_ftrs = self.layers[self.ftr_layer_idx]['outputs']
        
        Cnt_time = 0
        Cnt_time3 = 0
        
        sampSize = 53 # sample size
        respSize = 17 # response: output of ConvNet
        batchSize = 128
        Rms1_sum = 0 # for bicubic 
        Rms2_sum = 0 # for proposed 
        frame_num = 1000 # number of batches fetched 
        
        # video  = cv2.VideoWriter('video_HighRes.avi', -1, 20, (464, 368)) # size designed for vxid compression
        video  = cv2.VideoWriter('video_HighRes.avi', -1, 24, (477, 371)) # size designed for uncompressed
        if video.isOpened():
            print 'Video is initialized successfully!'
        video2  = cv2.VideoWriter('video_LowRes.avi', -1, 24, (241, 188)) 
        if video2.isOpened():
            print 'Video2 is initialized successfully!'
        
        threadLock = threading.Lock() # to synchronize list operations
        if isinstance( next_data[3][0], list): # frames are captured from video files
            t = threading.Thread( target=show_image, args=(self.orinList, self.lresList, self.hresList, self.bicuList, threadLock,))
        else: # frames are captured from camera            
            t = threading.Thread( target=show_camera_image, args=(self.orinList, self.hresList, self.bicuList, threadLock,))

        t.start()
            
        Start2 = time.time()
        for iout in range( frame_num):
            data = next_data[2]
            ftrs = zeros((data[0].shape[1], num_ftrs), dtype=single)
            
            self.libmodel.startFeatureWriter(data + [ftrs], self.ftr_layer_idx)           
            Start4 = time.time()
            if iout > 0: # show frames while the current one is computing
                Start_show = time.time()
                ftrs1 = ftrs0[:, 0:respSize*respSize*25]
                Start3 = time.time() ### 
                fea0  = reshape( ftrs1, [batchSize, respSize, respSize, 25], order='F')          
                End3 = time.time() ###
                fea1 = zeros((batchSize, sampSize, sampSize), dtype=single)
                      
                
                for i in range(respSize):
                    for j in range(respSize):
                        # fea1[:, i*3:i*3+5, j*3:j*3+5] += np.swapaxes( np.reshape( fea0[:, i, j, :], [128, 5, 5], order='F'), 1,2)
                        fea1[:, i*3:i*3+5, j*3:j*3+5] += swapaxes( reshape( fea0[:, i, j, :], [batchSize, 5, 5], order='F'), 1,2)
                            
                # fea1 = fea1[:, 4:-4, 4:-4]             
                # fea2 = fea1/d['CntMat']
                fea2 = fea1/d['CntMat_large']
                
                Cnt_time3 = Cnt_time3+End3-Start3 ###
                
                h_frame_list = []
                for i in range(2):
                    h_frame_list.append( zeros( [sampSize*7, sampSize*9], 'float32'))
                    # h_frame_list.append( np.zeros( [r_Size*8+8, r_Size*8+8], 'float32'))
                   
                for i in range( 7):
                    for j in range( 9):                                      
                        h_frame_list[0][i*sampSize:(i+1)*sampSize, j*sampSize:(j+1)*sampSize] += squeeze( fea2[i*9+j, :, :]).T
                        h_frame_list[1][i*sampSize:(i+1)*sampSize, j*sampSize:(j+1)*sampSize] += squeeze( fea2[64+i*9+j, :, :]).T
                        # h_frame_list[0][i*r_Size:(i+1)*r_Size+8, j*r_Size:(j+1)*r_Size+8] += np.squeeze( fea2[i*8+j, :, :]).T
                        # h_frame_list[1][i*r_Size:(i+1)*r_Size+8, j*r_Size:(j+1)*r_Size+8] += np.squeeze( fea2[64+i*8+j, :, :]).T
               
                # h_frame_list[0] = h_frame_list[0][4:-4, 4:-4]/d['CntMat_frame'][4:-4, 4:-4]
                # h_frame_list[1] = h_frame_list[1][4:-4, 4:-4]/d['CntMat_frame'][4:-4, 4:-4]
                if isinstance( next_data0[3][0], list):
                    Rms1_sum, Rms2_sum = process_image(next_data0, h_frame_list, sampSize, self.orinList, self.lresList, self.hresList, self.bicuList, threadLock, Rms1_sum, Rms2_sum)   
                else:
                    # process_camera_image(next_data0, h_frame_list, sampSize, self.orinList, self.hresList, self.bicuList, threadLock)
                    # process_camera_image(next_data0, h_frame_list, sampSize, self.orinList, self.hresList, self.bicuList, threadLock, video)
                    process_camera_image(next_data0, h_frame_list, sampSize, self.orinList, self.hresList, self.bicuList, threadLock, video, video2)
                    # process_camera_image(next_data0, h_frame_list, sampSize, self.orinList, self.hresList, self.bicuList, threadLock, iout)
                    
            if len(self.orinList) > 100:
                print 'List is too long!'
                break
            # load the next batch while the current one is computing
            next_data2 = self.get_next_batch_video(train=False)
            
            End4 = time.time()
            
            self.finish_batch() 
            End2 = time.time()            
         
            ftrs0 = ftrs
            next_data0 = next_data
            next_data = next_data2
            Cnt_time = Cnt_time + End2-Start2
            
            print 'Main: %.3f ' % (End2-Start2)
            # print '%.3f ' % (End4-Start4)            
            Start2 = End2

        print 'FOR loop: %5.3f secs' % Cnt_time3
        print 'Internal: %5.3f secs' % (Cnt_time)   
              
        if isinstance( next_data0[3][0], list):
            Rms1 = Rms1_sum / frame_num /2
            Rms2 = Rms2_sum / frame_num /2
            print '%.4f %.4f    ' % (Rms1, Rms2)
        sys.stdout.flush()
        self.close_camera() # close video camera
      
        t.join() # wait for Thread to terminate
        cv2.destroyAllWindows()
        
        video.release()
        video2.release()
        
    def start(self):
        if self.verbose:
            print 'flag1'
            self.op.print_values()
        # if self.show_cost:
            # print 'flag2'
            # self.plot_cost()
        # if self.show_filters:
            # print 'flag3'
            # self.plot_filters()
        # if self.show_preds:
            # print 'flag4'
            # self.plot_predictions()
        if self.write_features:
            print 'flag5'
            self.do_write_features()
        # if self.write_pixel_proj:
            # print 'flag6'
            # self.do_write_pixel_proj()
        pl.show()
        End = time.time()
        print 'Total time: %5.3f secs' % (End-Start) 
        sys.exit(0)
            
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('data_path_train', 'data_path_test', 'dp_type_train', 'dp_type_test', 'gpu', 'img_provider_file', 'load_file', 'train_batch_range', 'test_batch_range', 'verbose'):
                op.delete_option(option)
        op.add_option("test-only", "test_only", BooleanOptionParser, "Test and quit?", default=1)
        op.add_option("show-cost", "show_cost", StringOptionParser, "Show specified objective function", default="")
        op.add_option("show-filters", "show_filters", StringOptionParser, "Show learned filters in specified layer", default="")
        op.add_option("input-idx", "input_idx", IntegerOptionParser, "Input index for layer given to --show-filters", default=0)
        op.add_option("cost-idx", "cost_idx", IntegerOptionParser, "Cost function return value index for --show-cost", default=0)
        op.add_option("no-rgb", "no_rgb", BooleanOptionParser, "Don't combine filter channels into RGB in layer given to --show-filters", default=False)
        op.add_option("yuv-to-rgb", "yuv_to_rgb", BooleanOptionParser, "Convert RGB filters to YUV in layer given to --show-filters", default=False)
        op.add_option("channels", "channels", IntegerOptionParser, "Number of channels in layer given to --show-filters (fully-connected layers only)", default=0)
        op.add_option("show-preds", "show_preds", StringOptionParser, "Show predictions made by given softmax on test set", default="")
        op.add_option("only-errors", "only_errors", BooleanOptionParser, "Show only mistaken predictions (to be used with --show-preds)", default=False, requires=['show_preds'])
        op.add_option("write-features", "write_features", StringOptionParser, "Write test data features from given layer", default="", requires=['feature-path'])
        op.add_option("feature-path", "feature_path", StringOptionParser, "Write test data features to this path (to be used with --write-features)", default="")
        op.add_option("write-pixel-proj", "write_pixel_proj", StringOptionParser, "Write the projection of some response on pixel space", default = "", requires=['response_idx'])
        op.add_option("multiview", "mult_view", IntegerOptionParser, "Number of views for multi-view testing", default=1)
        op.options['load_file'].default = None
        return op

def float2uint8( tmp1):
    l_idx = tmp1 < 0
    tmp1[l_idx] = 0
    h_idx = tmp1 > 255
    tmp1[h_idx] = 255
    tmp2 = uint8(rint(tmp1))
    return tmp2

def rmse(predictions, targets):
    return sqrt(((predictions - targets) ** 2).mean())
    
def process_image(next_data, h_frame_list, r_Size, OrinList, LresList, HresList, BicuList, threadLock, Rms1_sum, Rms2_sum):
    # Designed for processing images captured from video files
    Ofst = 2 # 6
    for i in range(2): 
        ori_frame_ycrcb = cv2.cvtColor(next_data[3][i][0], cv2.COLOR_BGR2YCR_CB)
        frame_ycrcb = cv2.cvtColor(next_data[3][i][1], cv2.COLOR_BGR2YCR_CB)
        upsc_frame1 = cv2.resize(next_data[3][i][1], (r_Size*9+Ofst*2, r_Size*7+Ofst*2), interpolation=cv2.INTER_CUBIC)
        upsc_frame_ycrcb = cv2.resize(frame_ycrcb, (r_Size*9+Ofst*2, r_Size*7+Ofst*2), interpolation=cv2.INTER_CUBIC)
        h_frame_ycrcb = zeros( [r_Size*7, r_Size*9, 3], dtype=uint8) 
        h_frame_list[i] = float2uint8( h_frame_list[i]) # Convert from float32 to uint8
        h_frame_ycrcb[:,:,0] = h_frame_list[i]               
        h_frame_ycrcb[:,:,1:3] = upsc_frame_ycrcb[Ofst:-Ofst, Ofst:-Ofst, 1:3]
        h_frame = cv2.cvtColor(h_frame_ycrcb, cv2.COLOR_YCR_CB2BGR) 
        # calculate RMSE & PSNR
        Rms1 = rmse(upsc_frame_ycrcb[Ofst:-Ofst, Ofst:-Ofst, 0], ori_frame_ycrcb[Ofst:-Ofst, Ofst:-Ofst, 0]) # for bicubic interpolation
        Rms2 = rmse(h_frame_list[i], ori_frame_ycrcb[Ofst:-Ofst, Ofst:-Ofst, 0]) # for high-res 
        # PSNR1 = 20*log10( 255/Rms1)
        # PSNR2 = 20*log10( 255/Rms2)
        # print '%.2f %.2f %.2f %.2f   ' % (Rms1, Rms2, PSNR1, PSNR2), 
        
        Rms1_sum += Rms1;
        Rms2_sum += Rms2;

        # Get lock to synchronize threads
        # threadLock.acquire()
        OrinList.append( next_data[3][i][0])
        LresList.append( next_data[3][i][1])
        HresList.append( h_frame)
        BicuList.append( upsc_frame1)
        # Free lock to release next thread
        # threadLock.release()
 
    return Rms1_sum, Rms2_sum

# def process_camera_image(next_data, h_frame_list, r_Size, OrinList, HresList, BicuList, threadLock, iout):
# def process_camera_image(next_data, h_frame_list, r_Size, OrinList, HresList, BicuList, threadLock, video):
def process_camera_image(next_data, h_frame_list, r_Size, OrinList, HresList, BicuList, threadLock, video, video2):
# def process_camera_image(next_data, h_frame_list, r_Size, OrinList, HresList, BicuList, threadLock):
    # Designed for processing images captured from camera
    Ofst = 2 # 6
    for i in range(2):       
        ori_frame_ycrcb = cv2.cvtColor(next_data[3][i], cv2.COLOR_BGR2YCR_CB)
        upsc_frame1 = cv2.resize(next_data[3][i], (r_Size*9+Ofst*2, r_Size*7+Ofst*2), interpolation=cv2.INTER_CUBIC)
        upsc_frame_ycrcb = cv2.resize(ori_frame_ycrcb, (r_Size*9+Ofst*2, r_Size*7+Ofst*2), interpolation=cv2.INTER_CUBIC)
        h_frame_ycrcb = zeros( [r_Size*7, r_Size*9, 3], dtype=uint8)   
        h_frame_list[i] = float2uint8( h_frame_list[i]) # Convert from float32 to uint8
        h_frame_ycrcb[:,:,0] = h_frame_list[i]               
        h_frame_ycrcb[:,:,1:3] = upsc_frame_ycrcb[Ofst:-Ofst, Ofst:-Ofst, 1:3]
        h_frame = cv2.cvtColor(h_frame_ycrcb, cv2.COLOR_YCR_CB2BGR) 
        
        # cv2.imwrite( './HighQualVideo/High_resFrame'+ str(iout*2+i) +'.png', h_frame)
        # cv2.imwrite( './HighQualVideo/Low_resFrame'+ str(iout*2+i) +'.png', next_data[3][i])
        # video.write( h_frame[1:1+368, 6:6+464])
        video.write( h_frame)
        video2.write( next_data[3][i])
        
        # Get lock to synchronize threads
        # threadLock.acquire()
        OrinList.append( next_data[3][i])
        HresList.append( h_frame)
        BicuList.append( upsc_frame1)
        # Free lock to release next thread
        # threadLock.release()     
    
    # print 'Length of OrinList = %d' % len(OrinList)
    
    return 
    
def show_image( OrinList, LresList, HresList, BicuList, threadLock):
    time.sleep(2)
    threadLock.acquire()
    if not OrinList:
        print 'It does not work!'
        return
    # else:
        # print 'Length: ', len(OrinList)
    threadLock.release()
    loopTime = 0.07 # 0.075
    cnt = 0    
    emp_cnt = 0
    Start_in = time.time()
    while (True):
        if not OrinList: # if there is no frame in the list, then wait for 0.1 sec!
            time.sleep( 0.1)
            emp_cnt += 1
            if emp_cnt > 10: # if waiting for more than 1 sec, break!
                break
            continue
        
        emp_cnt = 0
        
        cv2.imshow('Original', OrinList[0])
        cv2.imshow('Low-res Frame', LresList[0])
        cv2.imshow('High-res Frame', HresList[0])
        cv2.imshow('Bicubic Frame1', BicuList[0])
        del OrinList[0]
        del LresList[0]
        del HresList[0]
        del BicuList[0]

        # cv2.waitKey(1)  
        if ( cv2.waitKey(1) == ord('q')):
            break 

        cnt += 1  
        End_in0 = time.time()
        if loopTime-(End_in0-Start_in) > 0:
            time.sleep( loopTime-(End_in0-Start_in))           
            End_in0 = time.time()
            print 'Inside loop: %.3f ' % ( loopTime) 
        else:
            print 'Inside loop: %.3f ' % ( End_in0-Start_in) 

        Start_in = End_in0
        
    print 'Last frame #', (cnt)
    # threadLock.acquire()
    # print 'In the end, length = ', len(OrinList)
    # threadLock.release()
    return
        
def show_camera_image( OrinList, HresList, BicuList, threadLock):
    # Designed for processing images captured from camera
    time.sleep(2) # unit: sec
    threadLock.acquire()
    if not OrinList:
        print 'It does not work!'
        return
    threadLock.release()
    loopTime = 0.07 # 0.075
    cnt = 0    
    emp_cnt = 0
    Start_in = time.time()
    while (True):
        if not OrinList: # if there is no frame in the list, then wait for 0.1 sec!
            time.sleep( 0.1)
            emp_cnt += 1
            if emp_cnt > 10: # if waiting for more than 1 sec, break!
                break
            continue
        
        emp_cnt = 0        
        cv2.imshow('Original', OrinList[0])
        cv2.imshow('High-res Frame', HresList[0])
        cv2.imshow('Bicubic Frame1', BicuList[0])
        del OrinList[0]
        del HresList[0]
        del BicuList[0]
        if ( cv2.waitKey(1) == ord('q')):
            break        
      
        cnt += 1        
        End_in0 = time.time()
        if loopTime-(End_in0-Start_in) > 0:
            time.sleep( loopTime-(End_in0-Start_in))           
            End_in0 = time.time()
            print 'Inside loop: %.3f ' % ( loopTime) 
        else:
            print 'Inside loop: %.3f ' % ( End_in0-Start_in) 
        Start_in = End_in0
        
    print 'Last frame #', (cnt)
    return
     
    
if __name__ == "__main__":
    try:
        Start = time.time()
        op = ShowConvNet.get_options_parser()
        op, load_dic = IGPUModel.parse_options(op)
        # End0 = time.time()
        # print 'Before ShowConvNet: %5.3f secs' % (End0-Start)
        model = ShowConvNet(op, load_dic)
        # End1 = time.time()
        # print 'Before model.start: %5.3f secs' % (End1-Start)
        model.start()
        
    except (UnpickleError, ShowNetError, opt.GetoptError), e:
        print "----------------"
        print "Error:"
        print e 
