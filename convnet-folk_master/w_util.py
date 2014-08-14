import numpy as n

def readLines(fname):
    with open(fname, 'r') as fid:
        return fid.readlines()

def get_box(w, h, scale, crop):
    if crop[0]=='u':
        l=int(scale*min(w, h))
        x1 = w-l
        y1 = h-l
        crop = crop[1:].split('-')
        base_view = int(crop[0])
        j = int(crop[1])
        cidx = j%base_view
        ridx = j/base_view
        dx = int(x1*cidx/(base_view-1))
        dy = int(y1*ridx/(base_view-1))
        return n.array([dy+1, dx+1, dy+l, dx+l])
    elif crop[0]=='g': #g5x3-14
        l=int(scale*min(w, h)) #224/256, or 224/448
        x1 = w-l
        y1 = h-l
        crop = crop[1:].split('-')
        j = int(crop[1])
        grid = n.array([int(_) for _ in crop[0].split('x')])
        g1=n.max(grid)
        g2=n.min(grid)
        idx1=j%g1
        idx2=j/g1
        #print j, grid, idx1, idx2
        if w>h:
            dx=int(x1*idx1/(g1-1))
            dy=int(y1*idx2/(g2-1))
        else:
            dx=int(x1*idx2/(g2-1))
            dy=int(y1*idx1/(g1-1))
        #print w, h, dx, dy
        return n.array([dy+1, dx+1, dy+l, dx+l])
    else:
        print "unknown crop:", crop
        assert(0)

def merge_box(bpred):
    #bpred = n.median(bpred, axis=0).astype(int) #median
    #bpred = bpred.reshape((1, 4))
    m=bpred.shape[0]
    ov=n.zeros([m, m])-1000
    for i in range(bpred.shape[0]):
        for j in range(0, i):
            ov[i, j]=rect_overlap(bpred[i], bpred[j])[0]
            #ov[i, j]=-1*rect_dist(bpred[i], bpred[j])
    while (1):
        m=bpred.shape[0]
        i=n.argmax(ov)
        imax,jmax=n.unravel_index(i, (m,m))
        ovmax=ov[imax, jmax]
        if (ovmax<0.8):
            break
        else: #merge
            for i in range(m):
                if jmax<i:
                    ov[i, jmax]=rect_overlap(bpred[i], bpred[jmax])[0]
                    #ov[i, jmax]=-1*rect_dist(bpred[i], bpred[jmax])
                elif jmax>i:
                    ov[jmax, i]=rect_overlap(bpred[i], bpred[jmax])[0]
                    #ov[jmax, i]=-1*rect_dist(bpred[i], bpred[jmax])
            ov=n.delete(ov, imax, 0)
            ov=n.delete(ov, imax, 1)
            bpred[jmax]=(bpred[imax]+bpred[jmax])/2
            bpred=n.delete(bpred, imax, 0)

    return bpred

def rect_dist(rect1, rect2):
    dx=abs((rect1[2]-rect1[0])-(rect2[2]-rect2[0]))
    dy=abs((rect1[3]-rect1[1])-(rect2[3]-rect2[1]))
    return n.sqrt(dx*dx+dy*dy)

def rect_overlap(rect1, rect2):
    xmin1, ymin1, xmax1, ymax1 = rect1
    xmin2, ymin2, xmax2, ymax2 = rect2
    ixmin = max(xmin1, xmin2)
    ixmax = min(xmax1, xmax2)
    iymin = max(ymin1, ymin2)
    iymax = min(ymax1, ymax2)
    s1=(xmax1-xmin1+1)*(ymax1-ymin1+1)
    s2=(xmax2-xmin2+1)*(ymax2-ymin2+1)
    
    xoverlap = max(0, ixmax - ixmin + 1)
    yoverlap = max(0, iymax - iymin + 1)
    sov=xoverlap*yoverlap*1.0
    return sov/(s1+s2-sov), (ixmin, iymin, ixmax, iymax)

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
