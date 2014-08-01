# initialization of the weight from other model

import numpy as n
import scipy.io

from util import *

initFromModel = None

def makew(name, idx, shape, params=None):
    layerName = params[0]
    modelfile = params[1]
    pdic={'trnp': 0, 'fltr': -1}
    for i in range(2,len(params),2):
        pdic[params[i]]=params[i+1]
    pdic['trnp']=int(pdic['trnp'])
    pdic['fltr']=int(pdic['fltr'])
    #print 'makew pdic:', pdic

    initFromModel = unpickle(modelfile)
    layers = initFromModel['model_state']['layers']
    l = [layer for layer in layers if layer['name'] == layerName]
    if len(l) == 0:
        print  'Cannot find the layer ' + layerName + ' for initialization'
        raise Exception()

    if idx<len(l[0]['weights']):
        w=l[0]['weights'][idx]
    else:
        print 'input {} not found, using zero weights'.format(idx)
        w=n.zeros(shape, dtype='float32')
    if pdic['trnp'] != 0:
        w=w.T
    if pdic['fltr'] > w.shape[-1]:
        print 'append zeros to form %d filters' % pdic['fltr']
        assert(len(w.shape)==2)
        tmp=n.zeros([pdic['fltr'], w.shape[0]], dtype=w.dtype)
        tmp[0:w.shape[1]]=w.T
        w=tmp.T
    return w

# params: layer name; model file name; transpose
def makeb(name, shape, params=None):
    layerName = params[0]
    modelfile = params[1]
    pdic={'trnp': 0, 'fltr': -1}
    for i in range(2,len(params),2):
        pdic[params[i]]=params[i+1]
    pdic['trnp']=int(pdic['trnp'])
    pdic['fltr']=int(pdic['fltr'])
    #print 'makeb pdic:', pdic

    initFromModel = unpickle(modelfile)
    layers = initFromModel['model_state']['layers']
    l = [layer for layer in layers if layer['name'] == layerName]
    if len(l) == 0:
        print  'Cannot find the layer '+ layerName + ' for initialization'
        raise Exception()

    if pdic['trnp'] == 0:
        b=l[0]['biases']
    else:
        b=l[0]['biases'].T
    if pdic['fltr'] > b.shape[0]:
        print 'append zeros to form %d filters' % pdic['fltr']
        tmp=n.zeros([pdic['fltr'], ]+list(b.shape[1:]), dtype=b.dtype)
        tmp[0:b.shape[0]]=b
        b=tmp
    return b

# read from text file
def makewfile(name, idx, shape, params=None):
    modelfile = params[0]
    w=n.loadtxt(modelfile, dtype='float32')
    w=w.T
    return w

# read from mat file
def makewmat(name, idx, shape, params=None):
    varname = params[0]
    modelfile = params[1]
    d=scipy.io.loadmat(modelfile)
    w=n.array(d[varname].T, dtype='float32')
    #print 'weights read from mat file with size:', n.shape(w)
    if shape[1] > w.shape[-1]:
        print 'append zeros to form %d filters' % shape[1]
        assert(len(w.shape)==2 and w.shape[0]==shape[0])
        tmp=n.zeros(shape, dtype=w.dtype)
        tmp[:, 0:w.shape[1]]=w
        w=tmp
    return w




