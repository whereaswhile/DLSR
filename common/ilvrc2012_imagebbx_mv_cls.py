import sys
import os
import numpy as np
import scipy.misc
import scipy.io as sio
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
from w_util import getsubimg, rgbimg2vec, readLines, gray2vec, rgb2gray
import bcfstore as bcfs
import StringIO

def getStore(datadir):
    return ILVRC2012_Set(datadir)

class ILVRC2012_Set:
	def __init__(self, paramfile):
		print "ilvrc2012_imagebbx_mv_cls with paramfile:", paramfile
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
		if self.param['bbxfile']=='0':
			self.bbx=None
			print 'no bounding boxes provided, using full image'
		else:
			bb=sio.loadmat(self.param['bbxfile'])
			self.bbx=[bb['res'][0][i][0,0][0] for i in range(len(bb['res'][0]))]
			assert(self.bbx[0].shape[1]==4)
			print '%d bounding boxes read' % (len(self.bbx))
                if self.param['imgfile'][-4:]=='.bcf':
                    self.bcfstore=bcfs.bcf_store_file(self.param['imgfile'])
                    print "{} files found in bcf file".format(self.bcfstore.size())
                    meta=sio.loadmat(self.param['metafile'])
                    self.bcfList=meta['bcfidx'][0]
                    self.rawsize=meta['imsize'][0]
                    self.rawsize=[[_[0,0][0][0][0], _[0,0][1][0][0] ] for _ in self.rawsize] #[w, h]
                if self.param['imglist']!='-1':
                    self.imgList=readLines(self.param['imglist'])
                    self.imgList=[int(_.rstrip()) for _ in self.imgList]
                elif self.bbx!=None: 
                    self.imgList=range(1, 1+len(self.bbx)) #starts from 1
                elif self.param['imgfile'][-4:]=='.bcf':
                    self.imgList=range(1, 1+len(self.bcfList)) #starts from 1
                elif 'metafile' in self.param:
                    meta=sio.loadmat(self.param['metafile'])
                    self.imgList=range(1, 1+meta['imsize'].shape[1]) #starts from 1
                else:
                    print 'cannot parse imgList'
                    assert(0)
                self._readLabels(self.param['lblfile'])
		self.imgNum = len(self.imgList)
		#self.imgNum = 16
		print '%d images found' % self.imgNum
		self.curidx = -1 #globla index
		self.curimgidx = -1
		self.curimg = None
		self.curbbx = None

	def _readLabels(self, lfname):
		lines = readLines(lfname)
		self.labels = np.array([int(line) for line in lines])

	def get_num_images(self):
		return self.imgNum*len(self.pertcomb)

	def get_num_classes(self):
		return 1000

	def get_input_dim(self, idx):
                if idx==0: #image input
                    return self.param['imgsize']*self.param['imgsize']*3
                elif idx==1: #binary class label
                    return self.get_num_classes()
                else:
                    assert(0)
                    return 0

	def get_output_dim(self):
		return 4

	def get_num_inputs(self):
                return 2

	def get_inputs(self, idx):
		self.curidx = idx
		crop, scale = self.pertcomb[idx%len(self.pertcomb)]
		imgidx = self.imgList[idx/len(self.pertcomb)]
		#print 'idx=%d, imgidx=%d' % (idx, imgidx)
		if self.curimgidx==imgidx:
			img = self.curimg
			lbl = self.curlbl
		else:
			if (self.param['imgfile'][-4:]=='.bcf'):
                            img = scipy.misc.imread(StringIO.StringIO(self.bcfstore.get(self.bcfList[imgidx-1]-1)))
                            lbl = self.labels[self.bcfList[imgidx-1]-1]
                            #print 'load bcf: ', imgidx-1
                        else:
                            img = scipy.misc.imread(self.param['imgfile'].format(imgidx))
                            lbl = self.labels[imgidx-1]
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
			self.curlbl = lbl
			self.curimgidx = imgidx

		# crop image and find bbx
		h, w, c = img.shape
		if (self.bbx==None or self.param['test']==1):
			b = np.array([1, 1, h, w], dtype=float)
		else:
			b = np.array(self.bbx[imgidx-1][0], dtype=float)
                        if (self.param['imgfile'][-4:]=='.bcf'): # convert from raw coordinate
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
			crop = [int(_) for _ in crop[1:].split('-')]
			assert(crop[0]>1)
			cidx = crop[1]%crop[0]
			ridx = crop[1]/crop[0]
			l = int(scale*min(w, h))
                        if (self.bbx==None or self.param['test']==1): #all over the image
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
			dx = int(x0+(x1-x0)*cidx/(crop[0]-1))
			dy = int(y0+(y1-y0)*ridx/(crop[0]-1))
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

                clsvect = np.zeros([self.get_num_classes(), ])
                clsvect[lbl-1] = 1

		return [img-self.meanImg, clsvect]

	#output bounding box [cx offset, cy offset, width, height], in the transformed image plane
	def get_output(self, idx):
		if idx!=self.curidx:
                    self.get_inputs(idx)
		return self.curbbx

	def get_meta(self, idx):
		return None

def test(param):
	ts = ILVRC2012_Set(param)
	print "{} images in total".format(ts.get_num_images())
	for i in range(0,500000,50000):
            inputs=ts.get_inputs(i)
            print "i={}, inputs={}".format(i, inputs)
	print 'image shape:', np.shape(inputs[0]), 'class number: ', inputs[1].shape
	b = []
	for i in range(0, 16, 1):
            im = ts.get_inputs(i)[0]
            bbx = ts.get_output(i)
            print i, bbx[0], im.shape
            b += [bbx[0]]
            scipy.misc.imsave('./img/{}.png'.format(i), im)
	sio.savemat('./img/bbx.mat', {'bbx': b})

if __name__ == '__main__':
	print 'testing ilvrc2012_imagebbx_mv_cls.py!'
	assert(len(sys.argv)==2)
	test(sys.argv[1])

