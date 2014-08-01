import numpy as n

def readLines(fname):
    with open(fname, 'r') as fid:
        return fid.readlines()

def get_box(w, h, scale, crop):
    l=int(scale*min(w, h))
    bx = (1+w)/2.0
    by = (0+h)/2.0
    x0 = max(0, bx-l)
    x1 = min(w-l, bx)
    y0 = max(0, by-l)
    y1 = min(h-l, by)
    crop = crop[1:].split('-')
    base_view = int(crop[0])
    j = int(crop[1])
    cidx = j%base_view
    ridx = j/base_view
    dx = int(x0+(x1-x0)/(base_view-1)*cidx)
    dy = int(y0+(y1-y0)/(base_view-1)*ridx)
    return n.array([dy+1, dx+1, dy+l, dx+l])

# xmin1, ymin1, xmax1, ymax1 = rect1
def rect_overlap(r1, r2):
    s1=(r1[2]-r1[0]+1)*(r1[3]-r1[1]+1)
    s2=(r2[2]-r2[0]+1)*(r2[3]-r2[1]+1)
    ri=[max(r1[0], r2[0]), max(r1[1], r2[1]), min(r1[2], r2[2]), min(r1[3], r2[3])]
    
    sov=max(0, ri[2]-ri[0]+1)*max(0, ri[3]-ri[1]+1)*1.0
    return sov/(s1+s2-sov), ri

def rect_size(rect1):
    xmin, ymin, xmax, ymax = rect1
    return (xmax-xmin+1)*(ymax-ymin+1)

def getsubimg(img, rect):
    xmin1, ymin1, xmax1, ymax1 = rect
    if len(img.shape) == 3:
        return img[ymin1-1:ymax1, xmin1-1:xmax1, :]
    else:
        return img[ymin1-1:ymax1, xmin1-1:xmax1]

def img2vec(img, asimg=False):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return rgbimg2vec(img)
    elif len(img.shape) == 3 and img.shape[2]>1 and asimg==True:
        return n.concatenate(tuple([img[:, :, i].flatten() for i in range(img.shape[-1])]))
    else:
        return img.flatten()

def rgbimg2vec(img):
    rvec = img[:,:,0].flatten()
    gvec = img[:,:,1].flatten()
    bvec = img[:,:,2].flatten()
    vec = n.concatenate((rvec, gvec, bvec))
    return vec

def rgb2gray(rgb):
    r = rgb[:, :, 0] # slices are not full copies, they cost little memory
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    gray = (r*2220 + g*7067 + b*713) / 10000 # result is a 2D array
    return gray

def gray2vec(gray):
    chvec = gray.flatten()
    vec = n.concatenate((chvec, chvec, chvec))
    return vec
