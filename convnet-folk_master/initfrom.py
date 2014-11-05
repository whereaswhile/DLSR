# initialization of the weight from other model

import numpy as n
import scipy.io

from util import *

def foo(modelfile):
    print foo.initModelFile
foo.initModel = None
foo.initModelFile = None

def makew(name, idx, shape, params=None):
    layerName = params[0]
    modelfile = params[1]
    pdic={'trnp': 0, 'fltr': shape[1]}
    for i in range(2,len(params),2):
        pdic[params[i]]=params[i+1]
    pdic['trnp']=int(pdic['trnp'])
    pdic['fltr']=int(pdic['fltr'])
    #print 'makew pdic:', pdic

    if modelfile!=foo.initModelFile:
        foo.initModel = unpickle(modelfile)
        foo.initModelFile = modelfile
    layers = foo.initModel['model_state']['layers']
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
        print 'append values to form %d/%d filters' % (pdic['fltr']-w.shape[-1], pdic['fltr'])
        assert(len(w.shape)==2)
        tmp=n.zeros([pdic['fltr'], w.shape[0]], dtype=w.dtype)
        tmp[0:w.shape[1]]=w.T
        tmp[w.shape[1]:]=n.random.rand(pdic['fltr']-w.shape[1], w.shape[0])/10000.0
        w=tmp.T
    if shape[0] > w.shape[0]:
        print 'append values to form %d/%d signals' % (shape[0]-w.shape[0], shape[0])
        tmp=n.zeros(shape, dtype=w.dtype)
        tmp[0:w.shape[0], :]=w
        tmp[w.shape[0]:, :]=n.random.rand(shape[0]-w.shape[0], shape[1])/10000.0
        w=tmp
    return w

# params: layer name; model file name; transpose
def makeb(name, shape, params=None):
    layerName = params[0]
    modelfile = params[1]
    pdic={'trnp': 0, 'fltr': shape[1]}
    for i in range(2,len(params),2):
        pdic[params[i]]=params[i+1]
    pdic['trnp']=int(pdic['trnp'])
    pdic['fltr']=int(pdic['fltr'])
    #print 'makeb pdic:', pdic

    if modelfile!=foo.initModelFile:
        foo.initModel = unpickle(modelfile)
        foo.initModelFile = modelfile
    layers = foo.initModel['model_state']['layers']
    l = [layer for layer in layers if layer['name'] == layerName]
    if len(l) == 0:
        print  'Cannot find the layer '+ layerName + ' for initialization'
        raise Exception()

    if pdic['trnp'] == 0:
        b=l[0]['biases']
    else:
        b=l[0]['biases'].T
    if pdic['fltr'] > b.shape[-1]:
        print 'append values to form %d/%d filters' % (pdic['fltr']-b.shape[-1], pdic['fltr'])
	assert(b.shape[0]==1)
	assert(len(b.shape)==2)
        tmp=n.zeros([1, pdic['fltr']], dtype=b.dtype)
        tmp[0,0:b.shape[1]]=b
        tmp[0,b.shape[1]:]=n.random.rand(1, pdic['fltr']-b.shape[1])*0.0
        b=tmp
    return b

# read from text file
def makewfile(name, idx, shape, params=None):
    modelfile = params[0]
    w=n.loadtxt(modelfile, dtype='float32')
    w=w.T
    if len(w.shape)==1:
        w=w.reshape((len(w), 1))
        print 'reshape w to 2D:', w.shape
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




