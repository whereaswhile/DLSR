from data import *
from util import *
import numpy.random as nr
import numpy as n
import random as r
import scipy.misc
import imp

from w_util import getsubimg, rgbimg2vec, readLines, gray2vec, rgb2gray, img2vec

class ClassificationDataProvider(DataProvider):    
    def __init__(self, data_dir, batch_range=[0], init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test, read_meta=False)

        self.data_dir = data_dir
    
        # self.batchsize = 128
        self.batchsize = dp_params['minibatch_size']
    
        print 'Getting image provider from', dp_params['imgprovider']

        self.store = imp.load_source('neo', dp_params['imgprovider']).getStore(self.data_dir)

        self.numimgs = self.store.get_num_images()
        if self.numimgs % self.batchsize == 0:
            self.batch_range = range(self.numimgs / self.batchsize)
        else:
            self.batch_range = range(self.numimgs / self.batchsize + 1)

        self.init_batchnum = 0
        self.curr_batchnum = 0

        self.indexes = range(self.numimgs)

        if test:
            self.randseed = 0
        else:
            self.randseed = 1
            self.randgen = r.Random()
            self.randgen.seed(self.randseed)
            self.shuffle()

    def advance_batch(self):
        DataProvider.advance_batch(self)
        if not self.test and self.curr_batchnum == 0:
            self.shuffle()
    
    def shuffle(self):
        self.randgen.shuffle(self.indexes)
        
    def get_num_classes(self):
        return self.store.get_num_classes()

    def get_data_dims(self, idx=0):
        return self.store.get_data_dim() if idx == 0 else 1

    def get_batch(self, batch_num):
        st = batch_num * self.batchsize
        ed = min((batch_num + 1)*self.batchsize, self.numimgs)
        cursize = ed - st
        data = n.zeros((self.get_data_dims(), cursize), dtype = n.single)
        labels = n.zeros((1, cursize), dtype=n.single)
        for i in range(st, ed):
            img = self.store.get(self.indexes[i])            
            data[:, i % self.batchsize] = img2vec(img)
            labels[0, i % self.batchsize] = self.store.get_label(self.indexes[i])
        return [data, labels]
            
    
    def get_aux(self, batch_num):
        pass


class MultInstClsDataProvider(DataProvider):    
    def __init__(self, data_dir, batch_range=[0], init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test, read_meta=False)

        self.data_dir = data_dir
        # self.batchsize = 128
        self.batchsize = dp_params['minibatch_size']
    
        print 'class: MultInstClsDataProvider, getting image provider from', dp_params['imgprovider']

        self.store = imp.load_source('neo', dp_params['imgprovider']).getStore(self.data_dir)

        self.numimgs = self.store.get_num_images()
        if self.numimgs % self.batchsize == 0:
            self.batch_range = range(self.numimgs / self.batchsize)
        else:
            self.batch_range = range(self.numimgs / self.batchsize + 1)

        self.init_batchnum = 0
        self.curr_batchnum = 0

        self.indexes = range(self.numimgs)

        if test:
            self.randseed = 0
        else:
            self.randseed = 1
            self.randgen = r.Random()
            self.randgen.seed(self.randseed)
            self.shuffle()

    def advance_batch(self):
        DataProvider.advance_batch(self)
        if not self.test and self.curr_batchnum == 0:
            self.shuffle()
    
    def shuffle(self):
        self.randgen.shuffle(self.indexes)
        
    def get_num_classes(self):
        return self.store.get_num_classes()

    def get_data_dims(self, idx=0):
        return self.store.get_data_dim() if idx > 0 else 1

    def get_batch(self, batch_num):
        st = batch_num * self.batchsize
        ed = min((batch_num + 1)*self.batchsize, self.numimgs)
        cursize = ed - st

        labels = n.zeros((1, cursize), dtype=n.single)
        for i in range(st, ed):
            labels[0, i % self.batchsize] = self.store.get_label(self.indexes[i])

        inst_num=self.store.get_num_instances()            
        data = []
	for j in range(inst_num):
	    dj=n.zeros((self.get_data_dims(1), cursize), dtype = n.single)
            for i in range(st, ed):
        	img = self.store.get(j, self.indexes[i])
		#print type(img)
		#print n.shape(img)
        	dj[:, i % self.batchsize] = img2vec(img)
	    data += [dj]
        return [labels] + data
    
    def get_aux(self, batch_num):
        pass


class RegressionDataProvider(DataProvider):    
    def __init__(self, data_dir, batch_range=[0], init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test, read_meta=False)

        self.data_dir = data_dir
        # self.batchsize = 128
        self.batchsize = dp_params['minibatch_size']

        print 'class: RegDataProv, getting image provider from', dp_params['imgprovider']
        self.store = imp.load_source('neo', dp_params['imgprovider']).getStore(self.data_dir)

        self.numimgs = self.store.get_num_images()
        if self.numimgs % self.batchsize == 0:
            self.batch_range = range(self.numimgs / self.batchsize)
        else:
            self.batch_range = range(self.numimgs / self.batchsize + 1)

        self.init_batchnum = 0
        self.curr_batchnum = 0
        self.indexes = range(self.numimgs)

        if test:
            self.randseed = 0
        else:
            self.randseed = 1
            self.randgen = r.Random()
            self.randgen.seed(self.randseed)
            self.shuffle()

    def advance_batch(self):
        DataProvider.advance_batch(self)
        if not self.test and self.curr_batchnum == 0:
            self.shuffle()
    
    def shuffle(self):
        self.randgen.shuffle(self.indexes)
        
    #def get_num_classes(self): #not defined for regression
        #return self.store.get_num_classes()

    def get_data_dims(self, idx=0):
        if idx==0: #input data dimension
            return self.store.get_input_dim()
        else: #output data dimension
            return self.store.get_output_dim()

    def get_batch(self, batch_num):
        st = batch_num * self.batchsize
        ed = min((batch_num + 1)*self.batchsize, self.numimgs)
        cursize = ed - st
        data_in  = n.zeros((self.get_data_dims(0), cursize), dtype = n.single)
        data_out = n.zeros((self.get_data_dims(1), cursize), dtype = n.single)
        for i in range(st, ed):
            img = self.store.get_input(self.indexes[i])            
            data_in[:, i%self.batchsize] = img2vec(img, True) #use True for vectorize as image 
            img = self.store.get_output(self.indexes[i])            
            data_out[:, i%self.batchsize] = img2vec(img, True)
        return [data_in, data_out]

    def get_aux(self, batch_num):
        pass

class MultRegressionDataProvider(DataProvider):    
    def __init__(self, data_dir, batch_range=[0], init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test, read_meta=False)

        self.data_dir = data_dir
        # self.batchsize = 128
        self.batchsize = dp_params['minibatch_size']
        
        print 'class: MultRegDataProv, getting image provider from', dp_params['imgprovider']
        self.store = imp.load_source('neo', dp_params['imgprovider']).getStore(self.data_dir)

        self.numimgs = self.store.get_num_images()
        if self.numimgs % self.batchsize == 0:
            self.batch_range = range(self.numimgs / self.batchsize)
        else:
            self.batch_range = range(self.numimgs / self.batchsize + 1)

        self.init_batchnum = 0
        self.curr_batchnum = 0
        self.indexes = range(self.numimgs)

        if test:
            self.randseed = 0
        else:
            self.randseed = 1
            self.randgen = r.Random()
            self.randgen.seed(self.randseed)
            self.shuffle()

    def advance_batch(self):
        DataProvider.advance_batch(self)
        if not self.test and self.curr_batchnum == 0:
            self.shuffle()
    
    def shuffle(self):
        self.randgen.shuffle(self.indexes)
        
    def get_data_dims(self, idx=0):
	if idx==0: #output data dimension
            return self.store.get_output_dim()
	else: #input data dimension
            return self.store.get_input_dim(idx-1)

    def get_batch(self, batch_num):
        st = batch_num * self.batchsize
        ed = min((batch_num + 1)*self.batchsize, self.numimgs)
        cursize = ed - st

        data_out = n.zeros((self.get_data_dims(0), cursize), dtype = n.single)
        for i in range(st, ed):
            img = self.store.get_output(self.indexes[i])            
            data_out[:, i%self.batchsize] = img.flatten() 

        data_in = []
        input_num = self.store.get_num_inputs()
        for j in range(input_num):
            dj = n.zeros((self.get_data_dims(j+1), cursize), dtype=n.single)
            data_in += [dj]
        for i in range(st, ed):
            img = self.store.get_inputs(self.indexes[i])            
            for j in range(input_num):
                data_in[j][:, i%self.batchsize] = img2vec(img[j], True) #use True for vectorize as image 
        return [data_out] + data_in
    
    def get_aux(self, batch_num):
        pass

class VideoRegressionDataProvider(RegressionDataProvider):
    # Designed for real-time video super-resolution processing
    
    def get_batch(self, batch_num): # overwrite the old version in class "RegressionDataProvider"
        # used for the task of general video super-resolution 
        st = batch_num * self.batchsize
        ed = min((batch_num + 1)*self.batchsize, self.numimgs)
        cursize = ed - st
        data_in  = n.zeros((self.get_data_dims(0), cursize), dtype = n.single)
        data_out = n.zeros((self.get_data_dims(1), cursize), dtype = n.single)
        
        for i in range(st, ed):
            img = self.store.get_input( )  
            data_in[:, i%self.batchsize] = img2vec(img, True) #use True for vectorize as image 
            # img = self.store.get_output(self.indexes[i])            
            # data_out[:, i%self.batchsize] = img2vec(img, True)
        return [data_in, data_out]
    
    def get_batch_video(self, batch_num):
        # print 'Here is get_batch_video in imgdata.py'
        st = batch_num * self.batchsize
        ed = min((batch_num + 1)*self.batchsize, self.numimgs)
        cursize = ed - st # (128!!!)
        data_in  = n.zeros((self.get_data_dims(0), cursize), dtype = n.single)
        data_out = n.zeros((self.get_data_dims(1), cursize), dtype = n.single) # dummy output data
        img, data_ref = self.store.get_input()

        for i in range(st, ed):
            indx = self.indexes[i]            
            data_in[:, i%self.batchsize] = img2vec(img[indx], True) #use True for vectorize as image        
        return [data_in, data_out], data_ref
                 
    def get_next_batch_video(self):
        # print 'Here is get_next_batch_video in imgdata.py'
        self.data_dic, self.ref = self.get_batch_video(self.curr_batchnum) # get_batch_video is in imgdata.py
        epoch, batchnum = self.curr_epoch, self.curr_batchnum        
        return epoch, batchnum, self.data_dic, self.ref
        
    def close_camera(self):
        self.store.release_cam()
