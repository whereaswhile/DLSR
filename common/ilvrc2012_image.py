import sys
import os
import numpy as np
import scipy.misc
sys.path.append('../')
sys.path.append('../../common')

from w_util import getsubimg, rgbimg2vec, readLines, gray2vec, rgb2gray

def getStore(datadir):
    return ILVRC2012_Set(datadir)

class ILVRC2012_Set:
	def __init__(self, paramfile):
		print "ilvrc2012_image with paramfile:", paramfile
		self.param = {'paramfile': paramfile}
		plines=readLines(paramfile)
		for l in plines:
			l=l.rstrip().split()
			self.param[l[0]]=l[1]
		print self.param

		self.meanImg = np.load(self.param['meanimg'])
		self.meanImg = self.meanImg[16:256-16,16:256-16,:]
		self._readLabels(self.param['lblfile'])
		self.imgNum = int(self.param['imgnum'])

	def _readLabels(self, lfname):
		lines = readLines(lfname)
		self.labels = np.array([int(line) for line in lines])

	def get_num_images(self):
		return self.imgNum

	def get_num_classes(self):
		return 1000
        
	def get_data_dim(self):
		return 224*224*3
        
	def get(self, idx):
		img = scipy.misc.imread(self.param['imgfile'].format(idx+1))
		img = np.array(img)

		# convert to 3 channels
		if len(img.shape) == 2:
			newimg = np.zeros((img.shape)+(3,), dtype=img.dtype)
			newimg[:,:,0] = img
			newimg[:,:,1] = img
			newimg[:,:,2] = img
			img = newimg
		else:
			if img.shape[2] == 4:
				img = img[:,:,:3]

		# convert to 224x224
		h, w, c = img.shape
		ratio = 224.0 / min(w,h)
		img = scipy.misc.imresize(img, (max(224, int(h*ratio)), max(224, int(w*ratio))))
		img = getsubimg(img, (1, 1, 224, 224))

		return img-self.meanImg

	def get_label(self, idx):
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
	for i in range(0,200,10):
		im = ts.get(i)
		#print im.dtype, np.min(im), np.max(im)
		scipy.misc.imsave('./img/{}.png'.format(i), im.astype(float))

if __name__ == '__main__':
	print 'testing ilvrc2012_image.py!'
	assert(len(sys.argv)==2)
	test(sys.argv[1])

