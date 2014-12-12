# for training data bcf only

import sys
import os
import numpy as np
import scipy.misc
import scipy.io as sio
import cPickle as pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
from w_util import getsubimg, rgbimg2vec, readLines, gray2vec, rgb2gray, ismember
import bcfstore as bcfs
import StringIO

def getStore(datadir):
    return ILVRC2012_Set(datadir)

class ILVRC2012_Set:
	def __init__(self, paramfile):
		print "ilvrc2012_imagebbx_mv_bcfp with paramfile:", paramfile
		self.param = {'paramfile': paramfile, 'imgsize': 224, 'test': 0}
		plines=readLines(paramfile)
		for l in plines:
			l=l.rstrip().split()
			self.param[l[0]]=l[1]
		self.param['scale']=[float(_) for _ in self.param['scale'].split('+')]
		self.param['crop']=self.param['crop'].split('+') #list of initial strings
		self.param['imgsize']=int(self.param['imgsize'])
		self.param['test']=int(self.param['test'])
		assert(self.param['imgsize']==224)
		print self.param

		self.pertcomb=[]
		for crop in self.param['crop']:
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
					self.pertcomb+=[[c, s]]
					if c=='wh':
						break
		print 'image expanded with %d perturbation(s):' % len(self.pertcomb)
		print self.pertcomb
                sys.stdout.flush()

                if self.param['meanimg']=='-1':
                    print 'no meanimg specified, using 128 as mean'
                    self.meanImg = np.zeros([256, 256, 3])+128
                else:
                    self.meanImg = np.load(self.param['meanimg'])
		self.meanImg = self.meanImg[16:256-16,16:256-16,:]
		if self.param['imgsize']!=224:
			print 'reshape meanImg'
			self.meanImg = scipy.misc.imresize(np.round(self.meanImg).astype(np.uint8), \
                                                            (self.param['imgsize'], self.param['imgsize']))
			self.meanImg = self.meanImg.astype(float)

		#read label, bounding box, and index
		self.bcfstore=bcfs.bcf_store_file(self.param['imgfile'])
		print "{} files found in bcf file".format(self.bcfstore.size())
		bb=pickle.load(open(self.param['bbxfile'], 'rb'))
		self.bbx=bb['bbx']
		assert(self.bbx.shape[1]==4)
		print '%d bounding boxes read' % (self.bbx.shape[0])
		self.bcfList=bb['bcfidx']
		self.rawsize=bb['imsize'] #[w, h]

                if self.param['imglist']!='-1':
                    self.imgList=readLines(self.param['imglist'])
                    self.imgList=[int(_.rstrip()) for _ in self.imgList] #index in bcf
		    self.imgList=[np.where(self.bcfList==_)[0] for _ in self.imgList] #index in bbx, from 1
		    mask=np.array([len(_) for _ in self.imgList])
		    self.imgList=np.array(self.imgList)[mask==1]
		    self.imgList=[_[0]+1 for _ in self.imgList]
                else:
                    self.imgList=range(1, 1+self.bbx.shape[0]) #index in bbx, starts from 1
		self.imgNum = len(self.imgList)
		print '%d images found' % self.imgNum

		self.labels = np.zeros([max(self.imgList)+1, ])
		self.curidx = -1 #globla index
		self.curimgidx = -1
		self.curimg = None
		self.curbbx = None

	def get_num_images(self):
		return self.imgNum*len(self.pertcomb)

	def get_num_classes(self):
		return 1000

	def get_input_dim(self):
		return self.param['imgsize']*self.param['imgsize']*3

	def get_output_dim(self):
		return 4

	def get_input(self, idx):
		#print 'input idx:', idx
		self.curidx = idx
		crop, scale=self.pertcomb[idx%len(self.pertcomb)]
		imgidx = self.imgList[idx/len(self.pertcomb)] #image index in the 544539-bbx
		#print 'idx=%d, imgidx=%d' % (idx, imgidx)
		if self.curimgidx==imgidx:
			img = self.curimg
		else:
			img = scipy.misc.imread(StringIO.StringIO(self.bcfstore.get(self.bcfList[imgidx-1]-1)))
			#print 'load bcf: ', imgidx-1
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
			self.curimgidx = imgidx

		# crop image and find bbx
		h, w, c = img.shape
		if (self.param['test']==1):
			b = np.array([1, 1, h, w], dtype=float)
		else:
			b = self.bbx[imgidx-1].astype(float)
                        # convert from raw coordinate
                        s=self.rawsize[imgidx-1]
                        #print "image converted from", s, "to", (w, h)
                        b=b*w/s[0]
                        b[0]=max(1, b[0]);
                        b[1]=max(1, b[1]);
                        b[2]=min(h, b[2]);
                        b[3]=min(w, b[3]);
		if crop=='wh':
			l = min(w, h)
			dx = 0
			dy = 0
		elif crop[0]=='u': #uniform grid
			l = int(scale*min(w, h))
                        if (self.param['test']==1): #all over the image
                            x0 = 0
                            x1 = w-l
                            y0 = 0
                            y1 = h-l
                        else: #at least cover half of ground truth b
                            bx = (b[1]+b[3])/2.0
                            by = (b[0]+b[2])/2.0
                            x0 = max(0, bx-l)
                            x1 = min(w-l, bx)
                            y0 = max(0, by-l)
                            y1 = min(h-l, by)
			crop = [int(_) for _ in crop[1:].split('-')]
			cidx = crop[1]%crop[0]
			ridx = crop[1]/crop[0]
			assert(crop[0]>0)
			if (crop[0]>1):
			    dx = int(x0+(x1-x0)*cidx/(crop[0]-1))
			    dy = int(y0+(y1-y0)*ridx/(crop[0]-1))
			else: #(crop[0]==1)
			    dx = int(x0+(x1-x0)/2)
			    dy = int(y0+(y1-y0)/2)
		else:
			print 'undefined CROP: %s' % crop
			assert(0)
		img = getsubimg(img, (dx+1, dy+1, dx+l, dy+l) )

		# convert to imgsize [224x224]
		ll = self.param['imgsize']
		img = scipy.misc.imresize(img, (ll, ll))
		if self.param['test']==1:
			self.curbbx = np.array([[0, 0, 0, 0]])
		else:
			self.curbbx = (np.array([b.tolist()]) - np.array([[dy, dx, dy, dx]]))*(1.0*ll/l)
			self.curbbx = np.round(self.curbbx).astype(int)

		return img-self.meanImg

	#output bounding box [cx offset, cy offset, width, height], in the transformed image plane
	def get_output(self, idx):
		if idx!=self.curidx:
			self.get_input(idx)
		return self.curbbx

	def get_label(self, idx):
		return self.labels[self.imgList[idx/len(self.pertcomb)]-1]

	def get_meta(self, idx):
		return None

def test(param):
	ts = ILVRC2012_Set(param)
	print "{} images in total".format(ts.get_num_images())
	for i in range(0,500,500):
		im=ts.get_input(i)
		y=ts.get_label(i)
                print "i={}, label={}".format(i, y)
	print 'image shape:', np.shape(im)
	b = []
	for i in range(11000, 13000, 100):
		im = ts.get_input(i)
		bbx = ts.get_output(i)
		print i, bbx[0], im.shape
		b += [bbx[0]]
		scipy.misc.imsave('./img/{}.jpg'.format(i), im)
	sio.savemat('./img/bbx.mat', {'bbx': b})

if __name__ == '__main__':
	print 'testing ilvrc2012_imagebbx_mv_bcfp.py!'
	assert(len(sys.argv)==2)
	test(sys.argv[1])

