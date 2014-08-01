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

		self._readLabels(self.param['lblfile'])
		self.imgNum = int(self.param['imgnum'])
		self.imgSize = int(self.param['size'])
		self.meanImg = np.load(self.param['meanimg'])
		self.meanImg = self.meanImg[16:256-16,16:256-16,:]
		self.meanImg = scipy.misc.imresize(np.round(self.meanImg).astype(np.uint8), (self.imgSize, self.imgSize))
		self.meanImg = self.meanImg.astype(float)
		#print 'average image size:', self.meanImg.shape

	def _readLabels(self, lfname):
		lines = readLines(lfname)
		self.labels = np.array([int(line) for line in lines])

	def get_num_images(self):
		return self.imgNum

	def get_num_classes(self):
		return 1008 #multiple of 16
        
	def get_data_dim(self):
		return self.imgSize*self.imgSize*3
        
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

		# convert to self.imgSize
		h, w, c = img.shape
		l = self.imgSize
		if 0: #with zero padding on sides
			imgfull = np.zeros([l, l, c], dtype=self.meanImg.dtype)
			if h<=w:
				h1=int(round(1.0*l*h/w))
				img = scipy.misc.imresize(img, (h1, l))
				imgfull[(l-h1)/2:(l-h1)/2+h1, :, :] = img - self.meanImg[(l-h1)/2:(l-h1)/2+h1, :, :]
			else:
				w1=int(round(1.0*l*w/h))
				img = scipy.misc.imresize(img, (l, w1))
				imgfull[:, (l-w1)/2:(l-w1)/2+w1, :] = img - self.meanImg[:, (l-w1)/2:(l-w1)/2+w1, :]
			#print img.shape, imgfull.shape
			return imgfull
		elif 0: #simple crop
			ratio = 224.0 / min(w,h)
			img = scipy.misc.imresize(img, (max(224, int(h*ratio)), max(224, int(w*ratio))))
			img = getsubimg(img, (1, 1, 224, 224))
			return img-self.meanImg
		else: #center crop
			if w<h:
				img = img[(h-w)/2:(h-w)/2+w, :, :]
			elif w>h:
				img = img[:, (w-h)/2:(w-h)/2+h, :]
			img = scipy.misc.imresize(img, (l, l))
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
	for i in range(0,20,1):
		im = ts.get(i)
		print im.dtype
		scipy.misc.imsave('./img/{}.png'.format(i), im)

if __name__ == '__main__':
	print 'testing ilvrc2012_image.py!'
	assert(len(sys.argv)==2)
	test(sys.argv[1])

