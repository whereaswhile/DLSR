import sys
import os
import numpy as np
import scipy.misc
import scipy.io as sio
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
from w_util import getsubimg, rgbimg2vec, readLines, gray2vec, rgb2gray

def getStore(datadir):
    return ILVRC2012_Set(datadir)

class ILVRC2012_Set:
	def __init__(self, paramfile):
            self.centers=np.array([[1, 1], [-1, -1], [1, 3]])
            self.K=self.centers.shape[0]
            self.dim=self.centers.shape[1]

            print "cluster_kcls.py with paramfile:", paramfile
            self.param = {'paramfile': paramfile, 'test': 0}
            plines=readLines(paramfile)
            for l in plines:
                l=l.rstrip().split()
                self.param[l[0]]=l[1]
            self.smplNum = int(self.param['smplnum'])
            self.imgNum = self.smplNum*self.K
            print self.param
            print '%d images found' % self.imgNum
            self.data = np.zeros([self.imgNum, self.dim])
            for k in range(0, self.K):
                self.data[self.smplNum*k:self.smplNum*(k+1)]=np.repeat(self.centers[k:k+1], self.smplNum, axis=0)
            self.data = self.data + (np.random.random_sample((self.imgNum, self.dim))-0.5)/1.0
            self.curidx = -1

	def get_num_images(self):
		return self.imgNum

	def get_num_classes(self):
		return self.K

	def get_input_dim(self, idx):
                if idx==0: #data input
                    return self.dim
                elif idx==1: #binary class label
                    return self.get_num_classes()
                else:
                    assert(0)
                    return 0

	def get_output_dim(self):
		return 1

	def get_num_inputs(self):
                return 2

	def get_inputs(self, idx):
		#print 'idx=%d' % (idx, )
		self.curidx = idx
                lbl = idx/self.smplNum
                img = self.data[idx]
                clsvect = np.zeros([self.get_num_classes(), ])
                clsvect[lbl] = 1

		return [img, clsvect]

	def get_output(self, idx):
		return np.array([0])

	def get_meta(self, idx):
		return None

def test(param):
	ts = ILVRC2012_Set(param)
	print "{} images in total".format(ts.get_num_images())
	for i in range(0,300,40):
            inputs=ts.get_inputs(i)
            output=ts.get_output(i)
            print "i={}, inputs={}, output={}".format(i, inputs, output)
	print 'image shape:', np.shape(inputs[0]), 'class number: ', inputs[1].shape

if __name__ == '__main__':
	print 'testing cluster_kcls.py!'
	assert(len(sys.argv)==2)
	test(sys.argv[1])

