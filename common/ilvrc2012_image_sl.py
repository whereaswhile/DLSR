import sys
import os
import numpy as np
import scipy.misc
import scipy.io as sio
sys.path.append('../')
sys.path.append('../../common')

from w_util import getsubimg, rgbimg2vec, readLines, gray2vec, rgb2gray

def getStore(datadir):
    return ILVRC2012_Set(datadir)

class ILVRC2012_Set:
	def __init__(self, paramfile):
		print "ilvrc2012_image_sl with paramfile:", paramfile
		self.param = {'paramfile': paramfile}
		plines=readLines(paramfile)
		for l in plines:
			l=l.rstrip().split()
			self.param[l[0]]=l[1]
		self.param['ncrop']=int(self.param['ncrop'])
		self.bbxpredf = self.param['ncrop']<0
		if self.bbxpredf:
			self.param['ncrop']=1
		self.param['scale']=float(self.param['scale'])
		print self.param

		self._readLabels(self.param['lblfile'])
		self.meanImg = np.load(self.param['meanimg'])
		self.meanImg = self.meanImg[16:256-16,16:256-16,:]
		self.imgNum = int(self.param['imgnum'])*self.param['ncrop']
		print "total expanded images: %d" % self.imgNum
		self.curidx = -1
		self.curimg = None

		bb=sio.loadmat(self.param['bbxfile'])
		self.bbx=[bb['res'][0][i][0,0][0] for i in range(bb['i'][0,0]-1)]
		print '%d bbx read' % (len(self.bbx))
		#self.bbx=[bb['res'][0][i][0,0][0][1:] for i in range(bb['i'][0,0]-1)] #remove the first whole box
		assert(len(self.bbx)>=int(self.param['imgnum']))
		assert(self.bbx[0].shape[1]==4)

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
		pertid = idx%self.param['ncrop']
		idx = idx/self.param['ncrop']
		if self.curidx==idx:
			img = self.curimg
		else:   
			img = scipy.misc.imread(self.param['imgfile'].format(idx+1))
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

		# crop image
		h, w, c = img.shape
		l=int(self.param['scale']*min(w, h)) #minimum required size
		if self.param['ncrop']>1:
			extra_bbx = pertid-self.bbx[idx].shape[0]
		elif self.bbxpredf: #using provided bbx
			extra_bbx = 200
			bb = self.bbx[idx][0]
		else: #['ncrop']=1  find the pre-defined crop with most overlap with bbx
			mask=np.zeros([h, w])
			mask[h/2, w/2]=1
			for i in range(self.bbx[idx].shape[0]-1):
				bb = self.bbx[idx][i+1]
				mask[(bb[0]-1):bb[2], (bb[1]-1):bb[3] ] = 1 #mask[(bb[0]-1):bb[2], (bb[1]-1):bb[3] ] + 1
			#scipy.misc.imsave('./img/mask_{}.png'.format(idx), mask)
			ovlp=np.zeros([10])
			ovlp[0] = np.mean(getsubimg(mask, (int(w-l)/2, int(h-l)/2, w-int(w-l)/2, h-int(h-l)/2) ))
			#ovlp[1] = np.mean(getsubimg(mask, (int(w-l)/3, int(h-l)/3, w-int(w-l)/3*2, h-int(h-l)/3*2) ))
			#ovlp[2] = np.mean(getsubimg(mask, (int(w-l)/3, int(h-l)/3*2, w-int(w-l)/3*2, h-int(h-l)/3) ))
			#ovlp[3] = np.mean(getsubimg(mask, (int(w-l)/3*2, int(h-l)/3, w-int(w-l)/3, h-int(h-l)/3*2) ))
			#ovlp[4] = np.mean(getsubimg(mask, (int(w-l)/3*2, int(h-l)/3*2, w-int(w-l)/3, h-int(h-l)/3) ))
			ovlp[1] = np.mean(getsubimg(mask, (1, int(h-l)/2, l, h-int(h-l)/2) ))
			ovlp[2] = np.mean(getsubimg(mask, (w-l+1, int(h-l)/2, w, h-int(h-l)/2) ))
			ovlp[3] = np.mean(getsubimg(mask, (int(w-l)/2, 1, w-int(w-l)/2, l)))
			ovlp[4] = np.mean(getsubimg(mask, (int(w-l)/2, h-l+1, w-int(w-l)/2, h)))
			ovlp[5] = np.mean(getsubimg(mask, (1, 1, l, l) ))
			ovlp[6] = np.mean(getsubimg(mask, (w-l+1, 1, w, l)))
			ovlp[7] = np.mean(getsubimg(mask, (1, h-l+1, l, h) ))
			ovlp[8] = np.mean(getsubimg(mask, (w-l+1, h-l+1, w, h) ))
			extra_bbx = np.argmax(ovlp)
			#print extra_bbx
		if extra_bbx<0: #use bbx
			bb = self.bbx[idx][pertid]
			if bb[3]-bb[1]<l-1:
				dl=(l-(bb[3]-bb[1]))/2
				if dl>=bb[1]:
					bb[1]=1
					bb[3]=l
				elif dl>=w-bb[3]:
					bb[3]=w
					bb[1]=w-l+1
				else:
					bb[1]-=dl
					bb[3]+=dl
			if bb[2]-bb[0]<l:
				dl=(l-(bb[2]-bb[0]))/2
				if dl>=bb[0]:
					bb[0]=1
					bb[2]=l
				elif dl>=h-bb[2]:
					bb[2]=h
					bb[0]=h-l+1
				else:
					bb[0]-=dl
					bb[2]+=dl
			#print "with bbx1:", bb
			img = getsubimg(img, (bb[1], bb[0], bb[3], bb[2]) )
		elif extra_bbx==0: #crop=='cn':
			img = getsubimg(img, (int(w-l)/2, int(h-l)/2, w-int(w-l)/2, h-int(h-l)/2) )
		elif extra_bbx==1: #crop=='cl':
			img = getsubimg(img, (1, int(h-l)/2, l, h-int(h-l)/2) )
		elif extra_bbx==2: #crop=='cr':
			img = getsubimg(img, (w-l+1, int(h-l)/2, w, h-int(h-l)/2) )
		elif extra_bbx==3: #crop=='tc':
			img = getsubimg(img, (int(w-l)/2, 1, w-int(w-l)/2, l))
		elif extra_bbx==4: #crop=='bc':
			img = getsubimg(img, (int(w-l)/2, h-l+1, w-int(w-l)/2, h))
		elif extra_bbx==5: #crop=='tl':
			img = getsubimg(img, (1, 1, l, l) )
		elif extra_bbx==6: #crop=='tr':
			img = getsubimg(img, (w-l+1, 1, w, l))
		elif extra_bbx==7: #crop=='bl':
			img = getsubimg(img, (1, h-l+1, l, h) )
		elif extra_bbx==8: #crop=='br':
			img = getsubimg(img, (w-l+1, h-l+1, w, h) )
		elif extra_bbx==100:
			img = getsubimg(img, (int(w-l)/2, int(h-l)/2, w-int(w-l)/2, h-int(h-l)/2) )
		elif extra_bbx==101:
			img = getsubimg(img, (int(w-l)/3, int(h-l)/3, w-int(w-l)/3*2, h-int(h-l)/3*2) )
		elif extra_bbx==102:
			img = getsubimg(img, (int(w-l)/3, int(h-l)/3*2, w-int(w-l)/3*2, h-int(h-l)/3) )
		elif extra_bbx==103:
			img = getsubimg(img, (int(w-l)/3*2, int(h-l)/3, w-int(w-l)/3, h-int(h-l)/3*2) )
		elif extra_bbx==104:
			img = getsubimg(img, (int(w-l)/3*2, int(h-l)/3*2, w-int(w-l)/3, h-int(h-l)/3) )
		elif extra_bbx==200:
			bb[0] = max(bb[0], 1)
			bb[1] = max(bb[1], 1)
			bb[2] = max(bb[2], bb[0]+1)
			bb[3] = max(bb[3], bb[1]+1)
			img = getsubimg(img, (bb[1], bb[0], bb[3], bb[2]) )

			w=bb[3]-bb[1]+1
			h=bb[2]-bb[0]+1 
			l=max(w, h)
			if 0: #warp to squre
				img = scipy.misc.imresize(img, (l, l))
			elif 1: #zero padding to square
				fullimg = scipy.misc.imresize(np.round(self.meanImg).astype(np.uint8), (l, l))
				if w>h:
					fullimg[(w-h)/2:(w-h)/2+h, :, :]=img
				elif h>w:
					fullimg[:, (h-w)/2:(h-w)/2+w, :]=img
				img = fullimg
		else:
			print 'exceeding extra bbx limit!'
			assert(0)

		# convert to 224x224
		h, w, c = img.shape
		if min(w, h)==0:
			print bb
		ratio = 224.0 / min(w,h)
		img = scipy.misc.imresize(img, (max(224, int(h*ratio)), max(224, int(w*ratio))))
		img = getsubimg(img, (1, 1, 224, 224))

		return img-self.meanImg

	def get_label(self, idx):
		return self.labels[idx/self.param['ncrop']]

	def get_meta(self, idx):
		return None

def test(param):
	ts = ILVRC2012_Set(param)
	print "{} images in total".format(ts.get_num_images())
	for i in range(0,200,10):
		#im=ts.get(i)
		meta=ts.get_meta(i)
		y=ts.get_label(i)
	print "i={}, label={}".format(i, y)
	#print 'image shape:', np.shape(im)
	print 'meta', meta
	for i in range(0,10,1):
		im = ts.get(i)
		scipy.misc.imsave('./img/{}.png'.format(i), im)

if __name__ == '__main__':
	print 'testing ilvrc2012_image.py!'
	assert(len(sys.argv)==2)
	test(sys.argv[1])

