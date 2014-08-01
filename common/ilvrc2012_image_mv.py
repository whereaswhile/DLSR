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
		print "ilvrc2012_image_mv with paramfile:", paramfile
		self.param = {'paramfile': paramfile, 'imgsize': 224}
		plines=readLines(paramfile)
		for l in plines:
			l=l.rstrip().split()
			self.param[l[0]]=l[1]
		self.param['scale']=[float(_) for _ in self.param['scale'].split('+')]
		self.param['crop']=self.param['crop'].split('+') #list of initial strings
		self.param['imgsize']=int(self.param['imgsize'])
		self.param['imgnum']=int(self.param['imgnum'])
		print self.param

		self.pertcomb=[]
		for crop in self.param['crop']:
			if crop[0]=='f':
				flip=1
				crop=crop[1:]
			else:
				flip=0
			if crop[0]=='u':
				nc=int(crop[1:])
				crops=['u{}-{}'.format(int(np.sqrt(nc)), _) for _ in range(nc)]
			elif crop[0:4]=='scan':
				nc=int(crop[4:])
				crops=['s{}-{}'.format(nc, _) for _ in range(nc) ]
			else:
				crops=[crop]
			for c in crops:
				for s in self.param['scale']:
					self.pertcomb+=[[c, flip, s]]
					if c=='wh':
						break
		print 'image expanded with %d perturbation(s):' % len(self.pertcomb)
		print self.pertcomb

		self.meanImg = np.load(self.param['meanimg'])
		self.meanImg = self.meanImg[16:256-16,16:256-16,:]
		self.meanImg = scipy.misc.imresize(np.round(self.meanImg).astype(np.uint8), \
											(self.param['imgsize'], self.param['imgsize']))
		self.meanImg = self.meanImg.astype(float)

		self._readLabels(self.param['lblfile'])
		self.imgNum = self.param['imgnum']*len(self.pertcomb)
		self.curidx = -1
		self.curimg = None

	def _readLabels(self, lfname):
		if lfname=='0':
			self.labels = np.ones(self.param['imgnum'], dtype=np.int)
		else:
			lines = readLines(lfname)
			self.labels = np.array([int(line) for line in lines])

	def get_num_images(self):
		return self.imgNum

	def get_num_classes(self):
		return 1008 #1000
        
	def get_data_dim(self):
		return self.param['imgsize']*self.param['imgsize']*3

	def get(self, idx):
		crop, flip, scale=self.pertcomb[idx%len(self.pertcomb)]
		idx = idx/len(self.pertcomb)
		if self.curidx==idx:
			img = self.curimg
		else:   
			img = scipy.misc.imread(self.param['imgfile'].format(idx+1)) #image index starting from 1
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
			self.curimg = img
			self.curidx = idx

		# flip left right
		if flip==1:
			img[:,:,0] = img[:, ::-1, 0]
			img[:,:,1] = img[:, ::-1, 1]
			img[:,:,2] = img[:, ::-1, 2]

		# crop image
		h, w, c = img.shape
		l=int(scale*min(w, h))
		if crop=='cn':
			img = getsubimg(img, (int(w-l)/2, int(h-l)/2, w-int(w-l)/2, h-int(h-l)/2) )
		elif crop=='tl':
			img = getsubimg(img, (1, 1, l, l) )
		elif crop=='tr':
			img = getsubimg(img, (w-l+1, 1, w, l))
		elif crop=='bl':
			img = getsubimg(img, (1, h-l+1, l, h) )
		elif crop=='br':
			img = getsubimg(img, (w-l+1, h-l+1, w, h) )
		elif crop=='tc':
			img = getsubimg(img, (int(w-l)/2, 1, w-int(w-l)/2, l))
		elif crop=='bc':
			img = getsubimg(img, (int(w-l)/2, h-l+1, w-int(w-l)/2, h))
		elif crop=='cl':
			img = getsubimg(img, (1, int(h-l)/2, l, h-int(h-l)/2) )
		elif crop=='cr':
			img = getsubimg(img, (w-l+1, int(h-l)/2, w, h-int(h-l)/2) )
		elif crop=='c1':
			img = getsubimg(img, (w-int(w-l)/4-l+1, int(h-l)/4, w-int(w-l)/4, int(h-l)/4+l-1) )
		elif crop=='c2':
			img = getsubimg(img, (w-int(w-l)/4-l+1, h-int(h-l)/4-l+1, w-int(w-l)/4, h-int(h-l)/4) )
		elif crop=='c3':
			img = getsubimg(img, (int(w-l)/4, h-int(h-l)/4-l+1, int(w-l)/4+l-1, h-int(h-l)/4) )
		elif crop=='c4':
			img = getsubimg(img, (int(w-l)/4, int(h-l)/4, int(w-l)/4+l-1, int(h-l)/4+l-1) )
		elif crop=='wh':
			img = img;
		elif crop[0]=='u': #uniform grid
			crop = [int(_) for _ in crop[1:].split('-')]
			assert(crop[0]>1)
			cidx = crop[1]%crop[0]
			ridx = crop[1]/crop[0]
			dx = (w-l)*cidx/(crop[0]-1)
			dy = (h-l)*ridx/(crop[0]-1)
			img = getsubimg(img, (dx+1, dy+1, dx+l, dy+l) )
		elif crop[0]=='s': #linear scan
			crop = [int(_) for _ in crop[1:].split('-')]
			if w<h:
				dy = (h-w)/(crop[0]-1)*crop[1]
				img = getsubimg(img, (1, dy+1, w, dy+w) )
			elif w>h:
				dx = (w-h)/(crop[0]-1)*crop[1]
				img = getsubimg(img, (dx+1, 1, dx+h, h) )
		else:
			print 'undefined CROP: %s, using whole image' % crop
			img = img;

		# convert to imgsize [224x224]
		h, w, c = img.shape
		l = self.param['imgsize']
		ratio = 1.0*l / min(w,h)
		img = scipy.misc.imresize(img, (max(l, int(h*ratio)), max(l, int(w*ratio))))
		img = getsubimg(img, (1, 1, l, l))
		if l!=224:
			self.meanImg

		return img-self.meanImg

	def get_label(self, idx):
		return self.labels[idx/len(self.pertcomb)]

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
		scipy.misc.imsave('./img/{}.png'.format(i), im)

if __name__ == '__main__':
	print 'testing ilvrc2012_image.py!'
	assert(len(sys.argv)==2)
	test(sys.argv[1])

