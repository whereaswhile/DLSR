import sys
import os
import numpy as np
import scipy.misc
import scipy.io as sio
import random as rnd
sys.path.append('../')
sys.path.append('../../common')
from w_util import getsubimg, rgbimg2vec, readLines, gray2vec, rgb2gray
import bcfstore as bcfs
import StringIO

def getStore(datadir):
    return ILVRC2012_Set(datadir)

class ILVRC2012_Set:
	def __init__(self, paramfile):
		print "ilvrc2012_joint with paramfile:", paramfile
                self.randgen = rnd.Random()
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

                self.bcfstore=bcfs.bcf_store_file(self.param['imgfile'])
		self.imgNum = self.bcfstore.size()
                print "{} files found in bcf file".format(self.imgNum)
                self.imgList=range(1, 1+self.imgNum) #starts from 1

		#read label, bounding box, and index
                print 'loading {}...'.format(self.param['lblfile'])
                self._readLabels(self.param['lblfile'])
                if self.param['bbxfile']=='0':
                    print 'no bbxfile'
                    self.bbx=None
                else:
                    print 'loading {}...'.format(self.param['bbxfile'])
                    bb=sio.loadmat(self.param['bbxfile'])
                    self.bbx=bb['bbx'][0] 
                    assert(self.bbx.shape[0]==self.imgNum)
                    print '%d bounding boxes read' % (len(self.bbx))
                    self.rawsize=bb['imsize'][0]

		self.curidx = -1 #globla index
		self.curimgidx = -1
		self.curimg = None
                sys.stdout.flush()

	def _readLabels(self, lfname):
		lines = readLines(lfname)
		self.labels = np.array([int(line) for line in lines])

	def get_num_images(self):
		return self.imgNum*len(self.pertcomb)

        def get_num_classes(self):
                return 1000

	def get_data_dim(self, idx):
                if idx==0: #image input
                    return self.param['imgsize']*self.param['imgsize']*3
                elif idx==1: #class label
                    return 1
                elif idx==2: #bbx
                    return 4
                elif idx==3: #binary given bbx
                    return 1
                else:
                    assert(0)
                    return 0

	def get_input_num(self):
                return 4

	def get_inputs(self, idx):
		self.curidx = idx
		crop, scale = self.pertcomb[idx%len(self.pertcomb)]
		imgidx = self.imgList[idx/len(self.pertcomb)]
		#print 'idx=%d, imgidx=%d' % (idx, imgidx)
		if self.curimgidx==imgidx:
                    img = self.curimg
		else:
                    img = scipy.misc.imread(StringIO.StringIO(self.bcfstore.get(imgidx-1)))
                    # convert to 3 channels
                    if len(img.shape) == 2:
                        newimg = np.zeros((img.shape)+(3,), dtype=img.dtype)
                        newimg[:,:,0] = img
                        newimg[:,:,1] = img
                        newimg[:,:,2] = img
                        img = newimg
                    elif img.shape[2] == 4:
                        img = img[:,:,:3]
                    self.curimg = img
                    self.curimgidx = imgidx
                lbl = self.labels[imgidx-1]

		# find bbx and crop image
		h, w, c = img.shape
                if self.bbx==None or self.bbx[imgidx-1].shape[1]==0: #no bbx provided, use whole image
                    bbxgiven=0
                    b = np.array([1, 1, h, w], dtype=float)
                else:
                    nb = self.bbx[imgidx-1].shape[0]
                    bbxgiven=-1
                    if nb==1: #has single bbx
                        b = np.array(self.bbx[imgidx-1][0], dtype=float)
                    else: #multiple bbx
                        ridx=range(nb)
                        self.randgen.shuffle(ridx)
                        b = np.array(self.bbx[imgidx-1][ridx[0]], dtype=float)
                    # scale from raw size to bcf size
                    s=self.rawsize[imgidx-1][0,0]
                    #print "image converted from", s, "to", (w, h)
                    s=[s[0][0][0], s[1][0][0]] #[w, h]
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
		img = getsubimg(img, (dx+1, dy+1, dx+l, dy+l))

		# convert to imgsize [224x224]
                #output bounding box [cx offset, cy offset, width, height], in the transformed image plane
		ll = self.param['imgsize']
		img = scipy.misc.imresize(img, (ll, ll))
		if self.param['test']==1:
                    curbbx = np.array([[0, 0, 0, 0]])
		else:
                    curbbx = (np.array([b.tolist()]) - np.array([[dy, dx, dy, dx]]))*(1.0*ll/l)
                    curbbx = np.round(curbbx).astype(int)

		return [img-self.meanImg, np.array([[lbl-1]]), curbbx, np.array([[bbxgiven]])]

	def get_meta(self, idx):
		return None

def test(param):
    ts = ILVRC2012_Set(param)
    print "{} images in total".format(ts.get_num_images())
    b=[]
    for i in range(680,704,1):
        inputs=ts.get_inputs(i)
        im = inputs[0]
        meta = inputs[1:4]
        print 'i={}, image shape:'.format(i), im.shape, 'meta: ', meta
        b += [meta[1]]
        scipy.misc.imsave('./img/{}.png'.format(i), im)
    sio.savemat('./img/bbx.mat', {'bbx': b})

if __name__ == '__main__':
	print 'testing ilvrc2012_joint.py!'
	assert(len(sys.argv)==2)
	test(sys.argv[1])

