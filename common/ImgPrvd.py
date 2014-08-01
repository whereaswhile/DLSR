import os
import sys
import numpy as np
import scipy.misc
import glob
sys.path.append("../convnet-folk_master")
from w_util import getsubimg, rgbimg2vec, readLines, gray2vec, rgb2gray

# define default parameters
CROP='cn'
CROP_PERC=0.5
PERT=0 #set perturb=1 will ignore all the CROP/CROP_PERC/FLIP...
PERT_CROP=['wh', 'cn', 'cl', 'cr', 'tl', 'tc', 'tr', 'bl', 'bc', 'br']
PERT_FLIP=[0, 1]
PERT_SCALE=[0.9, ] #[0.9, 0.8]

class ImgSet:
    def __init__(self, paramfile):
        
        print "ImgPrvd: parsing", paramfile
        plines = readLines(paramfile)
	self.param = {'crop': CROP, 'scale': CROP_PERC, 'imgpert': 0} #default param
	for l in plines:
	    l=l.rstrip().split()
	    self.param[l[0]]=l[1]
	self.param['scale']=float(self.param['scale'])
	self.param['flip']=(self.param['crop'][0]=='f')
	self.param['imgpert']=int(self.param['imgpert'])
	if self.param['flip']:
	    assert(len(self.param['crop'])==3)
	    self.param['crop']=self.param['crop'][1:]
	print self.param
	if self.param['imgpert']==1:
	    self.pertcomb=[]
	    for f in PERT_FLIP:
	        for c in PERT_CROP:
	            if(c=='wh'):
		        self.pertcomb+=[[c, f, 1]]
		    else:
		        for s in PERT_SCALE:
		            self.pertcomb+=[[c, f, s]]
	    #assert(len(self.pertcomb) == len(PERT_FLIP)+(len(PERT_CROP)-1)*len(PERT_FLIP)*len(PERT_SCALE))
	else:
	    self.pertcomb=[[self.param['crop'], self.param['flip'], self.param['scale']]]
	print 'image expanded with %d perturbation(s):' % len(self.pertcomb)
	print self.pertcomb
        self.fnames = [name.rstrip().strip('/') for name in readLines(self.param['imglist'])]
	self.cnames = [name.rstrip().strip('/') for name in readLines(self.param['clslist'])]
	clen = [len(name) for name in self.cnames]
	eidx = [fpath.rfind('/') for fpath in self.fnames]
	self.label = [0 for _ in range(len(self.fnames))]
	for i in range(len(self.label)):
	    fpath = self.fnames[i]  # eg. datapath/class/subclass/image.jpg
	    clsmtch = [fpath.find(c) for c in self.cnames]
	    clsmtch = [(clsmtch[j]!=-1) & (clsmtch[j]+clen[j]==eidx[i]) for j in range(len(clsmtch))]
	    assert(sum(clsmtch)==1)
	    self.label[i] = clsmtch.index(True)
	#for i in range(len(self.cnames)):
	#    assert(self.label.count(i)==50)
        self.meanImg = np.load(self.param['meanimg'])
        self.meanImg = self.meanImg[16:256-16,16:256-16,:]

    def get_num_images(self):
        return len(self.fnames)*len(self.pertcomb)
    
    def get_num_classes(self):
        return len(self.cnames)
        
    def get_data_dim(self):
        return 224*224*3

    def get(self, idx):
        crop, flip, scale=self.pertcomb[idx%len(self.pertcomb)]
        idx=idx/len(self.pertcomb)
        
        fpath = self.fnames[idx]
        img = scipy.misc.imread(os.path.join(self.param['imgfolder'], fpath))
	if idx%10000==0:
	    print "idx={0}, crop={1}, scale={2}, flip={3}".format(idx, crop, scale, flip)

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
	else:
	    #print 'undefined CROP, using all'
	    img = img;

	# convert to 224x224
        h, w, c = img.shape
        ratio = 224.0 / min(w,h)
        img = scipy.misc.imresize(img, (max(224, int(h*ratio)), max(224, int(w*ratio))))
        img = getsubimg(img, (1, 1, 224, 224))

        img = img - self.meanImg
        return img
        
    def get_label(self, idx):
        return self.label[idx/len(self.pertcomb)]
    
    def getmeta(self, idx):
        return self.fnames[idx/len(self.pertcomb)]
                

def getStore(param):
    return ImgSet(param)


def test(param):
    ts = ImgSet(param)
    print "{} images, {} classes".format(ts.get_num_images(), ts.get_num_classes())
    print ts.cnames[:10]
    for i in range(0,50,1):
        im=ts.get(i)
    print 'image shape:', np.shape(im)
    for i in range(0, ts.get_num_images(), 1):
        fn=ts.getmeta(i)
	y=ts.get_label(i)
	if i%100000==0:
            print "i={}, image={},\tlabel={}".format(i, fn, y)

if __name__ == '__main__':
    print 'testing ImgPrvd.py!'
    assert(len(sys.argv)==2)
    test(sys.argv[1])


