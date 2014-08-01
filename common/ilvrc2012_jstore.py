import sys
import os
import numpy as np
import scipy.misc
sys.path.append('../')
sys.path.append('../../common')

from jstore import Jstore, MemJstore
from w_util import getsubimg, rgbimg2vec, readLines, gray2vec, rgb2gray

def getStore(datadir):
    return ILVRC2012_Set(datadir)

class ILVRC2012_Set:
	def __init__(self, paramfile):
		print "ilvrc2012_jstore with paramfile:", paramfile
		self.param = {'paramfile': paramfile}
		plines=readLines(paramfile)
		for l in plines:
			l=l.rstrip().split()
			self.param[l[0]]=l[1]
		self.param['scale']=float(self.param['scale'])
		print self.param

		self.data_dir = self.param['imgfolder']
		self.store = MemJstore(self.param['imgfolder'])
		self.meanImg = np.load(self.param['meanimg'])
        
		self.meanImg = self.meanImg[16:256-16,16:256-16,:]
		self._readLabels(self.data_dir + '_labels.txt')

		self.stride=int(self.param['stride'])
		self.pertn=int((1-self.param['scale'])*256)/self.stride+1
		print "perturbation {}x{}".format(self.pertn, self.pertn)

	def _readLabels(self, lfname):
		lines = readLines(lfname)
		self.labels = np.array([int(line) for line in lines])

	def get_num_images(self):
		#return self.store.num_jpegs
		return 100*self.pertn*self.pertn

	def get_num_classes(self):
		return 1000
        
	def get_data_dim(self):
		return 224*224*3
        
	def get(self, idx):
		#print "image {} shape: {}".format(idx, self.store.get(idx).shape)
		#img = getsubimg(self.store.get(idx), (16, 16, 256-17, 256-17))

		pn=idx%(self.pertn*self.pertn)
		px=pn%self.pertn*self.stride
		py=pn/self.pertn*self.stride
		idx=idx/(self.pertn*self.pertn)
		img = self.store.get(idx)
		h, w, c = img.shape
		l = int(self.param['scale']*min(w, h))
		img = getsubimg(img, (1+px, 1+py, l+px, l+py))

		# convert to 224x224
		h, w, c = img.shape
		ratio = 224.0 / min(w,h)
		img = scipy.misc.imresize(img, (max(224, int(h*ratio)), max(224, int(w*ratio))))
		img = getsubimg(img, (1, 1, 224, 224))

		return img-self.meanImg

	def get_label(self, idx):
		idx=idx/(self.pertn*self.pertn)
		return self.labels[idx]

	def get_meta(self, idx):
		return None

def test(param):
	ts = ILVRC2012_Set(param)
	print "{} images in total".format(ts.get_num_images())
	for i in range(0,200,10):
		im=ts.get(i)
		meta=ts.get_meta(i)
		y=ts.get_label(i)
	print "i={}, label={}".format(i, y)
	print 'image shape:', np.shape(im)
	print 'meta', meta
	for i in range(100):
		im = ts.store.get(i)
		scipy.misc.imsave('./img/{}.png'.format(i), im)

if __name__ == '__main__':
	print 'testing ilvrc2012_jstore.py!'
	assert(len(sys.argv)==2)
	test(sys.argv[1])

