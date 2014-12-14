/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <helper_cuda.h>
#include <iostream>

#include <layer_kernels.cuh>
#include <layer.cuh>
#include <data.cuh>
#include <util.cuh>
#include <cudaconv2.cuh>
#include <matrix.h>

//#define DEBUG

using namespace std;


/* 
 * =======================
 * Layer
 * =======================
 */

Layer::Layer(ConvNet* convNet, PyObject* paramsDict, bool trans) : 
             _convNet(convNet),  _trans(trans) {
    _name = pyDictGetString(paramsDict, "name");
    _type = pyDictGetString(paramsDict, "type");
    
    _numGradProducersNext = 0;
    _foundGradConsumers = false;
    _gradConsumer = pyDictGetInt(paramsDict, "gradConsumer");
    _actsTarget = pyDictGetInt(paramsDict, "actsTarget");
    _actsGradTarget = pyDictGetInt(paramsDict, "actsGradTarget");
    _conserveMem = pyDictGetInt(paramsDict, "conserveMem");
    _outputs = _actsTarget < 0 ? new NVMatrix() : NULL;
    _actsGrad = _actsGradTarget < 0 ? new NVMatrix() : NULL;

    _dropout = pyDictGetFloat(paramsDict, "dropout");
    _dropout_mask = new NVMatrix();
}

void Layer::fpropNext(PASS_TYPE passType) {
    for (int i = 0; i < _next.size(); i++) {
        _next[i]->fprop(passType);
    }
}

void Layer::truncBwdActs() {
    // Only truncate actsGrad if I own it
    if (_conserveMem && _actsGradTarget < 0) { 
        getActsGrad().truncate();
    }
    if (_conserveMem) {
        getActs().truncate();
    }
}

void Layer::fprop(PASS_TYPE passType) {
    _rcvdFInputs += 1;

    //printf("fprop on layer %s\n", _name.c_str());
    Debug::print("Layer::fprop:" + _name, 1);
    if (_rcvdFInputs == _prev.size()) {
        NVMatrixV v;
        for (int i = 0; i < _prev.size(); i++) {
            v.push_back(&_prev[i]->getActs());
        }
        fprop(v, passType);
    }
}

void Layer::fprop(NVMatrix& v, PASS_TYPE passType) {
    NVMatrixV vl;
    vl.push_back(&v);
    fprop(vl, passType);
}

void Layer::fprop(NVMatrixV& v, PASS_TYPE passType) {
    assert(v.size() == _prev.size());
    _inputs.clear();
    _inputs.insert(_inputs.begin(), v.begin(), v.end());
    _outputs = _actsTarget < 0 ? _outputs : _inputs[_actsTarget];
    _rcvdFInputs = _prev.size();
    for (NVMatrixV::iterator it = v.begin(); it != v.end(); ++it) {
        (*it)->transpose(_trans);
    }
    getActs().transpose(_trans);
    
    // First do fprop on the input whose acts matrix I'm sharing, if any
    if (_actsTarget >= 0) {
        fpropActs(_actsTarget, 0, passType);
    }
    // Then add the rest of the inputs to that
    for (int i = 0; i < _prev.size(); i++) {
        if (i != _actsTarget) {
            fpropActs(i, _actsTarget >= 0 || i > 0, passType);
        }
    }

    dropout(passType);
    fpropNext(passType);
}

void Layer::dropout(PASS_TYPE passType) {
    if (passType != PASS_TEST && _dropout > 0.0) {
        _dropout_mask.resize(getActs().getNumRows(), getActs().getNumCols());
        _dropout_mask.randomizeUniform();
        _dropout_mask.biggerThanScalar(_dropout);
        getActs().eltwiseMult(_dropout_mask);
    }
      
    if (passType == PASS_TEST && _dropout > 0.0) {
        getActs().scale(1.0 - _dropout);
    }
}

void Layer::bprop(PASS_TYPE passType) {
    //printf("Bprop on layer %s\n", _name.c_str());
    Debug::print("Layer::bprop:" + _name, 1);
    if (_rcvdBInputs == _numGradProducersNext) {
        _rcvdBInputs++; // avoid doing bprop computation twice
        bprop(getActsGrad(), passType);
    }
}

void Layer::bprop(NVMatrix& v, PASS_TYPE passType) {
    Debug::print("Layer::bpropV:" + _name, 1);
    v.transpose(_trans);
    for (int i = 0; i < _prev.size(); i++) {
        _prev[i]->getActs().transpose(_trans);
        _prev[i]->getActsGrad().transpose(_trans);
    }
    getActs().transpose(_trans);
    
    if (_dropout > 0.0) {
      v.eltwiseMult(_dropout_mask);
    }

    bpropCommon(v, passType);
    
    if (isGradProducer()) {
        // First propagate activity gradient to all layers whose activity
        // gradient matrix I'm definitely not sharing.
        for (int i = 0; i < _prev.size(); i++) {
            if (_prev[i]->isGradConsumer() && _actsGradTarget != i) {
                bpropActs(v, i, _prev[i]->getRcvdBInputs() > 0 ? 1 : 0, passType);
                _prev[i]->incRcvdBInputs();
            }
        }
        // Then propagate activity gradient to the layer whose activity gradient
        // matrix I'm sharing, if any.
        if (_actsGradTarget >= 0 && _prev[_actsGradTarget]->isGradConsumer()) {
            bpropActs(v, _actsGradTarget, _prev[_actsGradTarget]->getRcvdBInputs() > 0 ? 1 : 0, passType);
            _prev[_actsGradTarget]->incRcvdBInputs();
        }
    }
    truncBwdActs();
    
    if (isGradProducer()) {
        for (int i = 0; i < _prev.size(); i++) {
            if (_prev[i]->isGradConsumer()) {
                _prev[i]->bprop(passType);
            }
        }
    }
}

void Layer::reset() {
    _rcvdFInputs = 0;
    _rcvdBInputs = 0;
}

string& Layer::getName() {
    return _name;
}

string& Layer::getType() {
    return _type;
}

int Layer::getRcvdFInputs() {
    return _rcvdFInputs;
}

int Layer::getRcvdBInputs() {
    return _rcvdBInputs;
}

int Layer::incRcvdBInputs() {
    return ++_rcvdBInputs;
}

void Layer::addNext(Layer* l) {
    _next.push_back(l);
    _numGradProducersNext += l->isGradProducer();
}

void Layer::addPrev(Layer* l) {
    _prev.push_back(l);
}

void Layer::postInit() {
//    _outputs = _actsTarget < 0 ? new NVMatrix() : &_prev[_actsTarget]->getActs();
    _actsGrad = _actsGradTarget < 0 ? new NVMatrix() : &_prev[_actsGradTarget]->getActsGrad();
}

// Does this layer, or some layer below it, need the gradient
// for parameter updates?
// Only weight layers should be grad consumers themselves.
bool Layer::isGradConsumer() {
    if (!_foundGradConsumers) {
        for (int i = 0; i < _prev.size(); i++) {
            _gradConsumer |= _prev[i]->isGradConsumer();
        }
        _foundGradConsumers = true;
    }
    return _gradConsumer;
}

// Does this layer produce gradient for layers below?
bool Layer::isGradProducer() {
    return true;
}

vector<Layer*>& Layer::getPrev() {
    return _prev;
}

vector<Layer*>& Layer::getNext() {
    return _next;
}

NVMatrix& Layer::getActs() {
    assert(_outputs != NULL);
    return *_outputs;
}

NVMatrix& Layer::getActsGrad() {
    assert(_actsGrad != NULL);
    return *_actsGrad;
}

void checkNaN(NVMatrix &mat, string name) {
    //    return ;
    Matrix *tmp = new Matrix();
    mat.copyToHost(*tmp, true);
    
    if (tmp->hasNan()) {
	printf("!!!! the matrix %s has NaN\n", name.c_str());
	//assert(false);
    }
	/*
    if (tmp->hasInf()) {
	printf("!!!! the matrix %s has Inf\n", name.c_str());
	assert(false);
    }
    */
	delete tmp;
}

/* 
 * =======================
 * NeuronLayer
 * =======================
 */
NeuronLayer::NeuronLayer(ConvNet* convNet, PyObject* paramsDict) 
    : Layer(convNet, paramsDict, true) {
    _neuron = &Neuron::makeNeuron(PyDict_GetItemString(paramsDict, "neuron"));
}

void NeuronLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
#ifdef DEBUG
	if (1) // && _name.compare("rms_neuron")==0) //debug
	{
		cout << "NeuronLayer::bpropActs, initial: " << _name << endl;
		cout << "v size: " << v.getNumRows() << ", " << v.getNumCols() << endl;
		cout << "v:" << endl;
		v.print(50, 1);
		getActsGrad().print(50, 1);
		_prev[0]->getActsGrad().print(50, 1);
		Layer *l=getNext()[0];
		for (int i=0; i<0; i++)
		{
			cout << "layer: " << l->getName() << endl;
			cout << "v size: " << l->getActsGrad().getNumRows() << ", " << l->getActsGrad().getNumCols() << endl;
			l->getActsGrad().print(10, 10);
			l=l->getNext()[0];
		}
	}
#endif
    _neuron->computeInputGrad(v, _prev[0]->getActsGrad(), scaleTargets > 0);
#ifdef DEBUG
	if (1) // && _name.compare("rms_neuron")==0) //debug
	{
		cout << _name << endl;
		_prev[0]->getActsGrad().print(4, 50);
		checkNaN(_prev[0]->getActsGrad(), "prev acts grad");
	}
#endif
}

void NeuronLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _neuron->activate(*_inputs[0], getActs());
#ifdef DEBUG
	if (0) //debug
	{
		cout << "NeuronLayer::fpropActs, " << _name << endl;
		_inputs[0]->print(2, 6);
		getActs().print(2, 6);
	}
#endif
}

/* 
 * =======================
 * WeightLayer
 * =======================
 */
WeightLayer::WeightLayer(ConvNet* convNet, PyObject* paramsDict, bool trans) : 
    Layer(convNet, paramsDict, trans) {
    string required[] = {"weights", "biases", "weightSourceLayerIndices", "weightSourceMatrixIndices"};
    pyCheckDict(paramsDict,required);

    // Source layers for shared weights
    intv& weightSourceLayerIndices = *pyDictGetIntV(paramsDict, "weightSourceLayerIndices");
    // Weight matrix indices (inside the above source layers) for shared weights
    intv& weightSourceMatrixIndices = *pyDictGetIntV(paramsDict, "weightSourceMatrixIndices");
  
    // for backward compatibility, set default value of weightType to "mom"
    string weightType;
    if (pyDictHas(paramsDict, "opttype")) 
	weightType = pyDictGetString(paramsDict, "opttype");
    else 
	weightType = "mom";

    MatrixV& hWeights = *pyDictGetMatrixV(paramsDict, "weights");
    Matrix& hBiases = *pyDictGetMatrix(paramsDict, "biases");

    if (weightType == "mom") {
	string required[] =  {"momW", "momB", "epsW", "epsB", "wc"};
        pyCheckDict(paramsDict, required);
        Matrix& hBiasesInc = *pyDictGetMatrix(paramsDict, "biasesInc");
        MatrixV& hWeightsInc = *pyDictGetMatrixV(paramsDict, "weightsInc");

        floatv& momW = *pyDictGetFloatV(paramsDict, "momW");
        float momB = pyDictGetFloat(paramsDict, "momB");
        floatv& epsW = *pyDictGetFloatV(paramsDict, "epsW");
        float epsB = pyDictGetFloat(paramsDict, "epsB");
        floatv& wc = *pyDictGetFloatV(paramsDict, "wc");

        for (int i = 0; i < weightSourceLayerIndices.size(); i++) {
            int srcLayerIdx = weightSourceLayerIndices[i];
            int matrixIdx = weightSourceMatrixIndices[i];
            if (srcLayerIdx == convNet->getNumLayers()) { // Current layer
                _weights.addWeights(*(new MomWeights(dynamic_cast<MomWeights&>(_weights[matrixIdx]), epsW[i])));
            } else if (srcLayerIdx >= 0) {
                WeightLayer& srcLayer = *static_cast<WeightLayer*>(&convNet->getLayer(srcLayerIdx));
                Weights* srcWeights = &srcLayer.getWeights(matrixIdx);
                _weights.addWeights(*new MomWeights(dynamic_cast<MomWeights&>(*srcWeights), epsW[i]));
            } else {
		//                _weights.addWeights(*new MomWeights(*hWeights[i], *hWeightsInc[i], epsW[i], wc[i], momW[i]));
                _weights.addWeights(*new MomWeights(*hWeights[i], *hWeightsInc[i], epsW[i], wc[i], momW[i]));
            }
        }

        _biases = new MomWeights(hBiases, hBiasesInc, epsB, 0, momB);

        delete &hWeightsInc;
        delete &momW;
        delete &epsW;
        delete &wc;
    }
    else if (weightType == "adagrad") {
	string required[] =  {"etaW", "etaB", "weightsGradAcc", "biasesGradAcc"};

        pyCheckDict(paramsDict, required);
        Matrix& hBiasesGradAcc = *pyDictGetMatrix(paramsDict, "biasesGradAcc");
        MatrixV& hWeightsGradAcc = *pyDictGetMatrixV(paramsDict, "weightsGradAcc");

        floatv& etaW = *pyDictGetFloatV(paramsDict, "etaW");
        float etaB = pyDictGetFloat(paramsDict, "etaB");

        for (int i = 0; i < weightSourceLayerIndices.size(); i++) {
            int srcLayerIdx = weightSourceLayerIndices[i];
            int matrixIdx = weightSourceMatrixIndices[i];
            if (srcLayerIdx == convNet->getNumLayers()) { // Current layer
                _weights.addWeights(*(new AdagradWeights(dynamic_cast<AdagradWeights&>(_weights[matrixIdx]))));
            } else if (srcLayerIdx >= 0) {
                WeightLayer& srcLayer = *static_cast<WeightLayer*>(&convNet->getLayer(srcLayerIdx));
                Weights* srcWeights = &srcLayer.getWeights(matrixIdx);
                _weights.addWeights(*new AdagradWeights(dynamic_cast<AdagradWeights&>(*srcWeights)));
            } else {
		//                _weights.addWeights(*new MomWeights(*hWeights[i], *hWeightsInc[i], epsW[i], wc[i], momW[i]));
                _weights.addWeights(*new AdagradWeights(*hWeights[i], *hWeightsGradAcc[i], etaW[i]));
            }
        }

        _biases = new AdagradWeights(hBiases, hBiasesGradAcc, etaB);

        delete &hWeightsGradAcc;
        delete &etaW;
    }

    else if (weightType == "adadelta") {
	string required[] =  {"epsW", "epsB", "rhoW", "rhoB", "weightsGradAcc", "biasesGradAcc", "weightsUpdateAcc", "biasesUpdateAcc"};

        pyCheckDict(paramsDict, required);
        Matrix& hBiasesGradAcc = *pyDictGetMatrix(paramsDict, "biasesGradAcc");
        MatrixV& hWeightsGradAcc = *pyDictGetMatrixV(paramsDict, "weightsGradAcc");
        Matrix& hBiasesUpdateAcc = *pyDictGetMatrix(paramsDict, "biasesUpdateAcc");
        MatrixV& hWeightsUpdateAcc = *pyDictGetMatrixV(paramsDict, "weightsUpdateAcc");

        floatv& epsW = *pyDictGetFloatV(paramsDict, "epsW");
        float epsB = pyDictGetFloat(paramsDict, "epsB");

        floatv& rhoW = *pyDictGetFloatV(paramsDict, "rhoW");
        float rhoB = pyDictGetFloat(paramsDict, "rhoB");

        for (int i = 0; i < weightSourceLayerIndices.size(); i++) {
            int srcLayerIdx = weightSourceLayerIndices[i];
            int matrixIdx = weightSourceMatrixIndices[i];
            if (srcLayerIdx == convNet->getNumLayers()) { // Current layer
                _weights.addWeights(*(new AdaDeltaWeights(dynamic_cast<AdaDeltaWeights&>(_weights[matrixIdx]))));
            } else if (srcLayerIdx >= 0) {
                WeightLayer& srcLayer = *static_cast<WeightLayer*>(&convNet->getLayer(srcLayerIdx));
                Weights* srcWeights = &srcLayer.getWeights(matrixIdx);
                _weights.addWeights(*new AdaDeltaWeights(dynamic_cast<AdaDeltaWeights&>(*srcWeights)));
            } else {
		//                _weights.addWeights(*new MomWeights(*hWeights[i], *hWeightsInc[i], epsW[i], wc[i], momW[i]));
                _weights.addWeights(*new AdaDeltaWeights(*hWeights[i], *hWeightsGradAcc[i], *hWeightsUpdateAcc[i], epsW[i], rhoW[i]));
            }
        }

        _biases = new AdaDeltaWeights(hBiases, hBiasesGradAcc, hBiasesUpdateAcc, epsB, rhoB);

        delete &hWeightsGradAcc;
        delete &epsW;
	delete &rhoW;
	delete &hWeightsUpdateAcc;
    }

    else{
        printf("!!! unknown weight type %s\n", weightType.c_str());
        assert(false);
    }

 
    

    

    // Epsilons for finite-difference gradient checking operation
    _wStep = 0.001;
    _bStep = 0.002;
    
    delete &weightSourceLayerIndices;
    delete &weightSourceMatrixIndices;
    delete &hWeights;

}

void WeightLayer::bpropCommon(NVMatrix& v, PASS_TYPE passType) {
    Debug::print("WeightLayer::bprop:" + _name, 1);
    if (_biases->isActive()) {
        bpropBiases(v, passType);
    }
    for (int i = 0; i < _weights.getSize(); i++) {
        if (_weights[i].isActive()) {
            bpropWeights(v, i, passType);
            // Increment its number of updates
            _weights[i].incNumUpdates();
        }
    }
    Debug::print("WeightLayer::bprop:exit:" + _name, 1);
}

void WeightLayer::updateWeights() {
    Debug::print("WeightLayer::updateWeights::weights::" + _name, 1);
    _weights.update();
    Debug::print("WeightLayer::updateWeights::biase::" + _name, 1);
    _biases->update();
}

void WeightLayer::copyToCPU() {
    _weights.copyToCPU();
    _biases->copyToCPU();
}

void WeightLayer::copyToGPU() {
    _weights.copyToGPU();
    _biases->copyToGPU();
}

void WeightLayer::checkGradients() {
    for (int i = 0; i < _weights.getSize(); i++) {
        _convNet->checkGradient(_name + " weights[" + tostr(i) + "]", _wStep, _weights[i]);
    }
    _convNet->checkGradient(_name + " biases", _bStep, *_biases);
}

Weights& WeightLayer::getWeights(int idx) {
    return _weights[idx];
}

/* 
 * =======================
 * FCLayer
 * =======================
 */
FCLayer::FCLayer(ConvNet* convNet, PyObject* paramsDict) : WeightLayer(convNet, paramsDict, true) {
    _wStep = 0.1;
    _bStep = 0.01;
}

void FCLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    getActs().addProduct(*_inputs[inpIdx], *_weights[inpIdx], scaleTargets, 1);
    if (scaleTargets == 0) {
        getActs().addVector(_biases->getW());
    }
}

void FCLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
#ifdef DEBUG
	if (1) //debug
	{   
		cout << "before FCLayer::bpropActs, " << _name << ", inpIdx=" << inpIdx << endl;
		v.printShape("v");
		v.print(64, 1);
		checkNaN(v, "v");
		_prev[inpIdx]->getActsGrad().printShape("prev acts grad");
	}
#endif
    NVMatrix& weights_T = _weights[inpIdx].getW().getTranspose();
    _prev[inpIdx]->getActsGrad().addProduct(v, weights_T, scaleTargets, 1);
    delete &weights_T;
#ifdef DEBUG
	if (1) //debug
	{   
		cout << "after FCLayer::bpropActs, " << _name << ", inpIdx=" << inpIdx << endl;
		_prev[inpIdx]->getActsGrad().printShape("prev acts grad");
		_prev[inpIdx]->getActsGrad().print(64, 3);
	}
#endif
}

void FCLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    //  printf("BpropBiasFC\n");
    int numCases = v.getNumRows();
    //    float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
    float scaleBGrad = 1.0 / numCases;
     _biases->getGrad().addSum(v, 0, 0, scaleBGrad);
}


void FCLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumRows();

    // printf("BpropWeightsFC %d\n", inpIdx);
    NVMatrix& prevActs_T = _prev[inpIdx]->getActs().getTranspose();
    //    float scaleInc = (_weights[inpIdx].getNumUpdates() == 0 && passType != PASS_GC) * _weights[inpIdx].getMom();
    float scaleOld = _weights[inpIdx].getNumUpdates() == 0? 0: 1;
    //  float scaleGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
    float scaleGrad = passType == PASS_GC ? 1 : 1.0 / numCases;

    _weights[inpIdx].getGrad().addProduct(prevActs_T, v, scaleOld, scaleGrad);
    
    //checkNaN(_weights[inpIdx].getGrad(), "fcbpropWeights_weight");
    delete &prevActs_T;

#ifdef DEBUG
	if (0) //debug
	{   
		cout << "FCLayer::bpropWeights, " << _name << ", inpIdx=" << inpIdx << endl;
		v.printShape("v");
		v.print(2, 6);
		checkNaN(v, "v");
		_prev[inpIdx]->getActs().print(2, 6);
		checkNaN(_prev[inpIdx]->getActs(), "prev acts");
		//_weights[inpIdx].getGrad().print(2, 6);
		//checkNaN(_weights[inpIdx].getGrad(), "w grad");
	}
#endif
}

/* 
 * =======================
 * LocalLayer
 * =======================
 */
LocalLayer::LocalLayer(ConvNet* convNet, PyObject* paramsDict) 
    : WeightLayer(convNet, paramsDict, false) {
    _padding = pyDictGetIntV(paramsDict, "padding");
    _stride = pyDictGetIntV(paramsDict, "stride");
    _filterSize = pyDictGetIntV(paramsDict, "filterSize");
    _channels = pyDictGetIntV(paramsDict, "channels");
    _imgSize = pyDictGetIntV(paramsDict, "imgSize");
    _numFilters = pyDictGetInt(paramsDict, "filters");
    _groups = pyDictGetIntV(paramsDict, "groups");
    _filterChannels = pyDictGetIntV(paramsDict, "filterChannels");
    _randSparse = pyDictGetIntV(paramsDict, "randSparse");
    _overSample = pyDictGetIntV(paramsDict, "overSample");
    _filterPixels = pyDictGetIntV(paramsDict, "filterPixels");
    _imgPixels = pyDictGetIntV(paramsDict, "imgPixels");
    
    _modulesX = pyDictGetInt(paramsDict, "modulesX");
    _modules = pyDictGetInt(paramsDict, "modules");

    // It's a vector on the heap to be consistent with all the others...
    _filterConns = new vector<FilterConns>();
    PyObject* pyFilterConns = PyDict_GetItemString(paramsDict, "filterConns");
    for (int i = 0; i < _randSparse->size(); i++) {
        FilterConns fc;
        if (_randSparse->at(i)) {
            fc.hFilterConns = getIntA(PyList_GET_ITEM(pyFilterConns, i));
        }
        _filterConns->push_back(fc);
    }
}

void LocalLayer::copyToGPU() {
    WeightLayer::copyToGPU();
    for  (int i = 0; i < _prev.size(); i++) {
        if (_randSparse->at(i)) { // Copy to GPU vector that describes sparse random connectivity
            cudaMalloc(&_filterConns->at(i).dFilterConns, sizeof(int) * _groups->at(i) * _filterChannels->at(i));
            cudaMemcpy(_filterConns->at(i).dFilterConns, _filterConns->at(i).hFilterConns,
                       sizeof(int) * _groups->at(i) * _filterChannels->at(i), cudaMemcpyHostToDevice);
            getLastCudaError("cudaMemcpy: failed");
        }
    }
}

/* 
 * =======================
 * ConvLayer
 * =======================
 */
ConvLayer::ConvLayer(ConvNet* convNet, PyObject* paramsDict) : LocalLayer(convNet, paramsDict) {
    _partialSum = pyDictGetInt(paramsDict, "partialSum");
    _sharedBiases = pyDictGetInt(paramsDict, "sharedBiases");
}

void ConvLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        convFilterActsSparse(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _filterConns->at(inpIdx).dFilterConns,
                             _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                             _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        convFilterActs(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx),
                       _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }
    
    if (scaleTargets == 0) {
        if (_sharedBiases) {
            getActs().reshape(_numFilters, getActs().getNumElements() / _numFilters);
            getActs().addVector(_biases->getW());
            getActs().reshape(_numFilters * _modules, getActs().getNumElements() / (_numFilters * _modules));
        } else {
            getActs().addVector(_biases->getW());
        }
    }
#ifdef DEBUG
	if (0) //debug
	{
		cout << "ConvLayer::fpropActs: " << _name << endl;
		_inputs[inpIdx]->print(2, 6);
		_weights[inpIdx].getW().print(2, 6);
		getActs().print(2, 6);
	}
#endif
}

void ConvLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    int numCases = v.getNumCols();
    //    float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
    float scaleBGrad = passType == PASS_GC ? 1 : 1.0 / numCases;
    if (_sharedBiases) {
        v.reshape(_numFilters, v.getNumElements() / _numFilters);
        _biases->getGrad().addSum(v, 1, 0, scaleBGrad);
        v.reshape(_numFilters * _modules, v.getNumElements() / (_numFilters * _modules));
    } else {
        _biases->getGrad().addSum(v, 1, 0, scaleBGrad);
    }
    //    checkNaN(_biases->getGrad(), "conv_bias_weight");
}

void ConvLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumCols();

    NVMatrix& tgt = _partialSum > 0 ? _weightGradTmp : _weights[inpIdx].getGrad();
    //    float scaleWGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
    //    float scaleTargets = _weights[inpIdx].getNumUpdates() > 0 && _partialSum == 0; // ? 1 : 0;
    float scaleWGrad = passType == PASS_GC ? 1 : 1.0 / numCases;
    float scaleTargets = _weights[inpIdx].getNumUpdates() > 0 && _partialSum == 0 ? 1 : 0;

    if (_randSparse->at(inpIdx)) {
        convWeightActsSparse(_prev[inpIdx]->getActs(), v, tgt, _filterConns->at(inpIdx).dFilterConns, _imgSize->at(inpIdx), _modulesX, _modulesX,
                             _filterSize->at(inpIdx), _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                             _filterChannels->at(inpIdx), _groups->at(inpIdx), _partialSum, scaleTargets, scaleWGrad);
    } else {
#ifdef DEBUG
		if (1) //debug
		{
			cout << "ConvLayer::bpropWeights-1, " << _name << ", inpIdx=" << inpIdx << endl;
			_prev[inpIdx]->getActs().printShape("prev acts");
			_prev[inpIdx]->getActs().print(30, 6);
			v.printShape("v");
			v.print(30, 6);
			//checkNaN(_prev[inpIdx]->getActs(), "prev acts");
			//checkNaN(v, "v");
			//checkNaN(_weights[inpIdx].getGrad(), "w grad");
		}
#endif
        convWeightActs(_prev[inpIdx]->getActs(), v, tgt, _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx), _padding->at(inpIdx),
                       _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), _partialSum, scaleTargets, scaleWGrad);
#ifdef DEBUG
		if (1) //debug
		{
			cout << "ConvLayer::bpropWeights0" << endl;
			_weights[inpIdx].getGrad().printShape("weight grad");
			_weights[inpIdx].getGrad().print(20, 6);
			//checkNaN(_weights[inpIdx].getGrad(), "w grad");
		}
#endif
    }

    if (_partialSum > 0) {
        scaleTargets = _weights[inpIdx].getNumUpdates() > 0;
        _weightGradTmp.reshape(_modules / _partialSum, _filterChannels->at(inpIdx) * _filterPixels->at(inpIdx) * _numFilters);
        _weights[inpIdx].getGrad().addSum(_weightGradTmp, 0, scaleTargets, 1);
        _weights[inpIdx].getGrad().reshape(_filterChannels->at(inpIdx) * _filterPixels->at(inpIdx), _numFilters);
    }

#ifdef DEBUG
	if (0) //debug
    {
		cout << "ConvLayer::bpropWeights: " << _name << endl;
		checkNaN(_weights[inpIdx].getGrad(), "conv_weight_weight");
		_weights[inpIdx].getGrad().print(2, 6);
	}
#endif
}

void ConvLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        NVMatrix& tgt = _overSample->at(inpIdx) > 1 ? _actGradTmp : _prev[inpIdx]->getActsGrad();
        convImgActsSparse(v, *_weights[inpIdx], tgt, _filterConns->at(inpIdx).dFilterConns,
                          _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX, _padding->at(inpIdx), _stride->at(inpIdx),
                          _channels->at(inpIdx), _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
        if (_overSample->at(inpIdx) > 1) {
            _actGradTmp.reshape(_overSample->at(inpIdx), _actGradTmp.getNumElements() / _overSample->at(inpIdx));
            _actGradTmp.sum(0, _prev[inpIdx]->getActsGrad());
            _prev[inpIdx]->getActsGrad().reshape(_prev[inpIdx]->getActsGrad().getNumElements() / v.getNumCols(), v.getNumCols());
        }
    } else {
        convImgActs(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(), _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
                    _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }

#ifdef DEBUG
	if (0) //debug
	{
		cout << "ConvLayer::bpropActs: " << _name << ", inpIdx=" << inpIdx << endl;
		v.print(10, 10);
		_prev[inpIdx]->getActsGrad().print(10, 10);
	}
#endif
}

void ConvLayer::truncBwdActs() {
    LocalLayer::truncBwdActs();
    if (_conserveMem) {
        _weightGradTmp.truncate();
        _actGradTmp.truncate();
    }
}
/* 
 * =======================
 * LocalUnsharedLayer
 * =======================
 */
LocalUnsharedLayer::LocalUnsharedLayer(ConvNet* convNet, PyObject* paramsDict) : LocalLayer(convNet, paramsDict) {
}

void LocalUnsharedLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        localFilterActsSparse(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _filterConns->at(inpIdx).dFilterConns,
                              _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                              _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        localFilterActs(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx),
                        _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);

    }  
    if (scaleTargets == 0) {
        getActs().addVector(_biases->getW());
    }
}

void LocalUnsharedLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    int numCases = v.getNumCols();
    float scaleBGrad = passType == PASS_GC ? 1 : 1.0 / numCases;
    _biases->getGrad().addSum(v, 1, 0, scaleBGrad);
}

void LocalUnsharedLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumCols();
    
    //    float scaleInc = (passType != PASS_GC && _weights[inpIdx].getNumUpdates() == 0) * _weights[inpIdx].getMom(); // momentum
    float scaleTargets = (passType != PASS_GC && _weights[inpIdx].getNumUpdates() > 0);

    float scaleWGrad = passType == PASS_GC ? 1 : 1.0 / numCases; 

    NVMatrix& tgt = _weights[inpIdx].getGrad();

    if (_randSparse->at(inpIdx)) {
        localWeightActsSparse(_prev[inpIdx]->getActs(), v, tgt, _filterConns->at(inpIdx).dFilterConns,
                              _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx), _padding->at(inpIdx), _stride->at(inpIdx),
                              _channels->at(inpIdx), _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, scaleWGrad);
    } else {
        localWeightActs(_prev[inpIdx]->getActs(), v, tgt, _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx),
                        _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, scaleWGrad);
    }
}

void LocalUnsharedLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        localImgActsSparse(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(), _filterConns->at(inpIdx).dFilterConns,
                           _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                           _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        localImgActs(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(),_imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
                    _padding->at(inpIdx),  _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }
}

/* 
 * =======================
 * SoftmaxLayer
 * =======================
 */
SoftmaxLayer::SoftmaxLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, true) {
}

void SoftmaxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& input = *_inputs[0];
    NVMatrix& max = input.max(1);
    input.addVector(max, -1, getActs());
    getActs().apply(NVMatrixOps::Exp());
    NVMatrix& sum = getActs().sum(1);
    getActs().eltwiseDivideByVector(sum);
    //    checkNaN(getActs(), "softmax output");
    delete &max;
    delete &sum;
}

void SoftmaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 0);
    bool doLogregGrad = _next.size() == 1 && _next[0]->getType() == "cost.logreg";
    if (doLogregGrad) {
        NVMatrix& labels = _next[0]->getPrev()[0]->getActs();
        float gradCoeff = dynamic_cast<CostLayer*>(_next[0])->getCoeff();
        computeLogregSoftmaxGrad(labels, getActs(), _prev[0]->getActsGrad(), scaleTargets == 1, gradCoeff);
    } else {
        computeSoftmaxGrad(getActs(), v, _prev[0]->getActsGrad(), scaleTargets == 1);
    }
}

/* 
 * =======================
 * EltwiseSumLayer
 * =======================
 */
EltwiseSumLayer::EltwiseSumLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _coeffs = pyDictGetFloatV(paramsDict, "coeffs");
	_dims = pyDictGetIntV(paramsDict, "dimensions"); //dimensions to select, -1 represent all
	_channels = pyDictGetInt(paramsDict, "channels");
	_imgSize = pyDictGetInt(paramsDict, "imgSize");
	_numPixels = pyDictGetInt(paramsDict, "imgPixels");
}

void EltwiseSumLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType)
{
	int dim=_dims->at(inpIdx);
#ifdef DEBUG
	if (0)
	{
	cout << "EltwiseSumLayer::fpropActs before\n";
	cout << "inpIdx: " << inpIdx << ", scaleTargets: " << scaleTargets << "\n";
	_inputs[inpIdx]->printShape("input size: ");
	}
#endif
    if (scaleTargets == 0)
	{
		if (dim==-1) //first input full size
		{
			_inputs[inpIdx]->scale(_coeffs->at(inpIdx), getActs());
		}
		else
		{
			getActs().reshape(_inputs[0]->getNumRows(), _inputs[0]->getNumCols());
			for (int c=0; c<_channels; c++)
			{
				_inputs[inpIdx]->sliceRows(dim*_numPixels, (dim+1)*_numPixels).scale(_coeffs->at(inpIdx),
					getActs().sliceRows(c*_numPixels, (c+1)*_numPixels) );
			}
		}
    }
	else
	{
		if (dim==-1) //input full size
		{
			getActs().add(*_inputs[inpIdx], _coeffs->at(inpIdx));
		}
		else
		{
			for (int c=0; c<_channels; c++)
			{
				getActs().sliceRows(c*_numPixels, (c+1)*_numPixels).add(
						_inputs[inpIdx]->sliceRows(dim*_numPixels, (dim+1)*_numPixels), _coeffs->at(inpIdx));
			}
#ifdef DEBUG
			if (0)//debug
			{
				cout << "EltwiseSumLayer::fpropActs, " << _name << ": inpIdx=" << inpIdx << endl;
				getActs().print(0, 4, 0, 6);
			}
#endif
		}
    }
}

void EltwiseSumLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType)
{
	int dim=_dims->at(inpIdx);
	if (dim==-1)
	{
		if (scaleTargets == 0)
		{
        	v.scale(_coeffs->at(inpIdx), _prev[inpIdx]->getActsGrad());
		}
		else
		{
			assert(&_prev[inpIdx]->getActsGrad() != &v);
			_prev[inpIdx]->getActsGrad().add(v, scaleTargets, _coeffs->at(inpIdx));
		}
	}
	else
	{
		if (scaleTargets == 0)
		{
			_inputs[inpIdx]->scale(0.0, _prev[inpIdx]->getActsGrad());
		}
		else
		{
			assert(&_prev[inpIdx]->getActsGrad() != &v);
		}
		_prev[inpIdx]->getActsGrad().scale(scaleTargets);
		for (int c=0; c<_channels; c++)
		{
			_prev[inpIdx]->getActsGrad().sliceRows(dim*_numPixels, (dim+1)*_numPixels).add(
		            v.sliceRows(c*_numPixels, (c+1)*_numPixels), _coeffs->at(inpIdx)); 
		}
	}
#ifdef DEBUG
	if (1) //debug
	{
		cout << "EltwiseSumLayer::bpropActs " << _name << ", inpIdx=" << inpIdx << ", dim=" << dim 
				<< ", scaleTargets=" << scaleTargets << ", _numPixels=" << _numPixels << endl;
		_inputs[inpIdx]->printShape("input size: ");
		cout << "===== actsgrad =====" << endl;
		_prev[inpIdx]->getActsGrad().printShape("prev acts grad");
		_prev[inpIdx]->getActsGrad().print(10, 10);
		checkNaN(_prev[inpIdx]->getActsGrad(), "prev acts grad");
		cout << "===== v =====" << endl;
		v.print(10, 4);
		checkNaN(v, "v");
	}
#endif
}

/* 
 * =======================
 * EltwiseProdLayer
 * =======================
 */

EltwiseProdLayer::EltwiseProdLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false)
{
	_coeffs = pyDictGetIntV(paramsDict, "dimensions"); //dimensions to select, -1 represent all
	_channels = pyDictGetInt(paramsDict, "channels");
	_imgSize = pyDictGetInt(paramsDict, "imgSize");
	_numPixels = pyDictGetInt(paramsDict, "imgPixels");
}

void EltwiseProdLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType)
{
    if (scaleTargets == 0)
	{
		assert(_coeffs->at(inpIdx)==-1); //first input always full size
        //_inputs[inpIdx]->scale(1, getActs());
        _inputs[inpIdx]->copy(getActs());
    }
	else
	{
		int coeff=_coeffs->at(inpIdx);
		if (coeff==-1)
		{
			getActs().eltwiseMult(*_inputs[inpIdx]);
		}
		else
		{
			//cout << "size=("  <<  _inputs[inpIdx]->getNumRows() << ", " << _inputs[inpIdx]->getNumCols() << ")\n";
			//cout << "c=" << _channels << ", i=" << _imgSize << ", p=" << _numPixels << endl;
			for (int c=0; c<_channels; c++)
			{
				getActs().sliceRows(c*_numPixels, (c+1)*_numPixels).eltwiseMult(
						_inputs[inpIdx]->sliceRows(coeff*_numPixels, (coeff+1)*_numPixels));
			}
		}
    }
#ifdef DEBUG
	if (0) //debug
	{
		cout << "==== EltwiseProdLayer::fpropActs, inpIdx=" << inpIdx << " ====\n";
		cout << "scaleTargets: " << scaleTargets << ", coeff: " << _coeffs->at(inpIdx) << "\n";
		cout << "input0\n";
		_inputs[0]->print(3, 20);
		cout << "input1\n";
		_inputs[1]->print(3, 20);
		getActs().print(0, 4, 0, 6);
		cout << long(_inputs[1]) << ", " << long(&getActs()) << "\n";
	}
#endif
}

void EltwiseProdLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType)
{
	int bsize=_inputs[inpIdx]->getNumCols();
	NVMatrix dact(_inputs[inpIdx]->getNumRows(), bsize);

	if (_prev.size()!=2)
	{
		cout << "EltwiseProdLayer::bpropActs only supports two inputs!\n";
		assert(false);
	}

	int cmpIdx = 1-inpIdx;
	int coeff = _coeffs->at(inpIdx);
	int cmpcoeff = _coeffs->at(cmpIdx);

    if (cmpcoeff==-1)
	{
		_inputs[cmpIdx]->copy(dact);
	}
	else
	{
		for (int c=0; c<_channels; c++)
		{
			_inputs[cmpIdx]->sliceRows(cmpcoeff*_numPixels, (cmpcoeff+1)*_numPixels).copy(
						dact.sliceRows(c*_numPixels, (c+1)*_numPixels));
		}
	}

#ifdef DEBUG
	cout << "==== EltwiseProdLayer::bpropActs, inpIdx=" << inpIdx << " ====\n";
	cout << "input0\n";
	_inputs[0]->print(3, 20);
	cout << "input1\n";
	_inputs[1]->print(3, 20);
	cout << "dact\n";
	dact.print(3, 20);
#endif

	dact.eltwiseMult(v); // multiply with gradents from later layers
    if (coeff==-1)
	{
		//nothing
    }
	else
	{
		for (int c=0; c<_channels; c++)
		{
			if (c==coeff) continue;
			dact.sliceRows(coeff*_numPixels, (coeff+1)*_numPixels).add(dact.sliceRows(c*_numPixels, (c+1)*_numPixels));
			dact.sliceRows(c*_numPixels, (c+1)*_numPixels).scale(0.f);
		}
    }

	// back propragate to previous input
    if (scaleTargets == 0 ) //first gradient back-prop to _prev[inpIdx]
	{
		dact.copy(_prev[inpIdx]->getActsGrad());
	}
	else
	{
		_prev[inpIdx]->getActsGrad().add(dact);
	}

#ifdef DEBUG
	if (1) //debug
	{
		cout << "previous layer size = " << _prev.size() << endl;
		cout << "coeff=" << coeff << endl;
		cout << "bsize=" << bsize << endl;
		_prev[inpIdx]->getActsGrad().printShape("prev grad size=");
		_prev[inpIdx]->getActsGrad().print(3, 64);
	}
#endif
}

/* 
 * =======================
 * EltwiseMaxLayer
 * =======================
 */
EltwiseMaxLayer::EltwiseMaxLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false){
}

void EltwiseMaxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (inpIdx == 1) { // the second input, max(*_inputs[0], *_inputs[1])
        _inputs[inpIdx]->applyBinary(NVMatrixAggs::Max(), *_inputs[0], getActs());
    } else if (inpIdx > 1) { //if more than 2 inputs
        getActs().applyBinary(NVMatrixAggs::Max(), *_inputs[inpIdx]);
    }
}

void EltwiseMaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    computeEltwiseMaxGrad(v, *_inputs[inpIdx], getActs(), _prev[inpIdx]->getActsGrad(), scaleTargets != 0);
}


/* 
 * =======================
 * AllMaxLayer 
 * =======================
 */
AllMaxLayer::AllMaxLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false){
}

void AllMaxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {

	_inputs[inpIdx]->max(0, getActs());
#ifdef DEBUG
	cout << "AllMaxLayer::fpropActs inpIdx=" << inpIdx << endl;
	_inputs[0]->printShape("input size:");
	_inputs[inpIdx]->print(0, 4, 0, 6);
	cout << "output:\n";
	getActs().print(0, 4, 0, 6);
#endif
}

void AllMaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    computeAllMaxGrad(v, *_inputs[inpIdx], getActs(), _prev[inpIdx]->getActsGrad(), scaleTargets != 0);
#ifdef DEBUG
    printf("AllMaxLayer::bpropActs prev[%d]->grad:\n", inpIdx);
	_prev[inpIdx]->getActsGrad().print(0, 4, 0, 6);
#endif
}

/* 
 * =======================
 * BoundBoxOverlapLayer
 * =======================
 */
BoundBoxOverlapLayer::BoundBoxOverlapLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false)
{
}

void BoundBoxOverlapLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType)
{
    if (inpIdx == 1) // Second input
	{
		int bsize=_inputs[0]->getNumCols();

		NVMatrix w0(1, bsize), h0(1, bsize), a0(1, bsize);
		NVMatrix w1(1, bsize), h1(1, bsize), a1(1, bsize);
		NVMatrix bi(4, bsize), wi(1, bsize), hi(1, bsize), ai(1, bsize);
		parseBoundBox(*_inputs[0], w0, h0, a0);
		parseBoundBox(*_inputs[1], w1, h1, a1);
		parseIntersectBoundBox(*_inputs[0], *_inputs[1], bi);
		parseBoundBox(bi, wi, hi, ai);
		
		a0.add(a1);
		a0.add(ai, -1);
		ai.eltwiseDivide(a0, getActs());
		getActs().scale(-1.f); //convert to cost: 1-overlap 
		getActs().addScalar(1.f);

#ifdef DEBUG
		if (0) //debug
		{
		cout << "\n==== fprop ====\n";
		_inputs[0]->print(0, 4, 0, 6);
		_inputs[1]->print(0, 4, 0, 6);
		cout << "ai/act:\n";
		ai.print(0, 1, 0, 6);
		getActs().print(0, 1, 0, 6);
		}
#endif
    }
}

void BoundBoxOverlapLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType)
{
	int bsize=v.getNumCols();
	NVMatrix w0(1, bsize), h0(1, bsize), a0(1, bsize);
	NVMatrix w1(1, bsize), h1(1, bsize), a1(1, bsize);
	NVMatrix bi(4, bsize), wi(1, bsize), hi(1, bsize), ai(1, bsize), u(1, bsize);
	NVMatrix dai(4, bsize), da(4, bsize), dact(4, bsize);

	parseBoundBox(*_inputs[0], w0, h0, a0);
	parseBoundBox(*_inputs[1], w1, h1, a1);
	parseIntersectBoundBox(*_inputs[0], *_inputs[1], bi);
	parseBoundBox(bi, wi, hi, ai);
	a0.add(a1, u);
	u.add(ai, -1);

	// find da
	NVMatrix *aw, *ah;
	if (inpIdx==0)
	{
		aw=&w0;
		ah=&h0;
	}
	else
	{
		aw=&w1;
		ah=&h1;
	}
	da.sliceRows(0, 1).add(*ah, 0, -1); //-h
	da.sliceRows(1, 2).add(*aw, 0, -1); //-w
	ah->copy(da.sliceRows(2, 3)); //h
	aw->copy(da.sliceRows(3, 4)); //w

	// find dai
	int cmpIdx=1-inpIdx;
	NVMatrix mask(1, bsize);
	ai.biggerThanScalar(0, mask); //nonzero mask
	wi.eltwiseMult(mask);
	hi.eltwiseMult(mask);
	dai.sliceRows(0, 1).add(hi, 0, -1); //-hi
	dai.sliceRows(1, 2).add(wi, 0, -1); //-wi
	hi.copy(dai.sliceRows(2, 3)); //hi
	wi.copy(dai.sliceRows(3, 4)); //wi
	_inputs[inpIdx]->biggerThan(*_inputs[cmpIdx], dact); //use dact temporally to store 0/1 binary
	dact.sliceRows(2, 4).addScalar(-1, 1, dact.sliceRows(2, 4));
	dai.eltwiseMult(dact);
	//cout << "\nbinary:\n"; dact.print(0, 4, 0, 6);

	// find dact
	dai.add(da, -1, dact);
	for (int i=0; i<4; i++)
	{
		dact.sliceRows(i, i+1).eltwiseMult(ai);
		dact.sliceRows(i, i+1).eltwiseDivide(u);
	}
	dact.add(dai);
	for (int i=0; i<4; i++)
	{
		dact.sliceRows(i, i+1).eltwiseDivide(u);
	}
	dact.scale(-1.f); //convert to cost: 1-overlap 

	// multiply with gradents from later layers
	for (int i=0; i<4; i++)
	{
		dact.sliceRows(i, i+1).eltwiseMult(v);
	}

#ifdef DEBUG
	if (0) //debug
	{
	cout << "\n==== bprop ====\n";
	cout << "inpIdx=" << inpIdx << "; ";
	cout << "v size: " << v.getNumRows() << ", " << v.getNumCols() << endl;
	cout << "act/v/dact\n";
	getActs().print(0, 1, 0, 6);
	v.print(0, 1, 0, 6);
	dact.print(0, 4, 0, 6);
	}
#endif

	// back propragate to previous input
    if (scaleTargets == 0 ) //first gradient back-prop to _prev[inpIdx]
	{
		dact.copy(_prev[inpIdx]->getActsGrad());
	}
	else
	{
		assert(_prev[inpIdx]->getActsGrad().getNumRows()==4);
		_prev[inpIdx]->getActsGrad().add(dact);
	}
}

/* 
 * =======================
 * DataLayer
 * =======================
 */
DataLayer::DataLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _dataIdx = pyDictGetInt(paramsDict, "dataIdx");
}

void DataLayer::fprop(PASS_TYPE passType) {
    throw string("No dava given!");
}

void DataLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
}

void DataLayer::fprop(NVMatrixV& data, PASS_TYPE passType) {
    data[_dataIdx]->copy(*_outputs);
    dropout(passType);
    fpropNext(passType);
}

bool DataLayer::isGradProducer() {
    return false;
}

/* 
 * =====================
 * PoolLayer
 * =====================
 */
PoolLayer::PoolLayer(ConvNet* convNet, PyObject* paramsDict, bool trans) 
    : Layer(convNet, paramsDict, trans) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _sizeX = pyDictGetInt(paramsDict, "sizeX");
    _start = pyDictGetInt(paramsDict, "start");
    _stride = pyDictGetInt(paramsDict, "stride");
    _outputsX = pyDictGetInt(paramsDict, "outputsX");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
    _pool = pyDictGetString(paramsDict, "pool");
}

PoolLayer& PoolLayer::makePoolLayer(ConvNet* convNet, PyObject* paramsDict) {
    string _pool = pyDictGetString(paramsDict, "pool");
    if (_pool == "max") {
        return *new MaxPoolLayer(convNet, paramsDict);
    } else if(_pool == "avg") {
        return *new AvgPoolLayer(convNet, paramsDict);
    }
    throw string("Unknown pooling layer type ") + _pool;
}

/* 
 * =====================
 * AvgPoolLayer
 * =====================
 */
AvgPoolLayer::AvgPoolLayer(ConvNet* convNet, PyObject* paramsDict) : PoolLayer(convNet, paramsDict, false) {
}

void AvgPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, AvgPooler());
}

void AvgPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalAvgUndo(v, _prev[0]->getActsGrad(), _sizeX, _start, _stride, _outputsX, _imgSize, scaleTargets, 1);
}

/* 
 * =====================
 * MaxPoolLayer
 * =====================
 */
MaxPoolLayer::MaxPoolLayer(ConvNet* convNet, PyObject* paramsDict) : PoolLayer(convNet, paramsDict, false) {
}

void MaxPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, MaxPooler());
}

void MaxPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalMaxUndo(_prev[0]->getActs(), v, getActs(), _prev[inpIdx]->getActsGrad(), _sizeX, _start, _stride, _outputsX, scaleTargets, 1);
}

/* 
 * =====================
 * NailbedLayer
 * =====================
 */
NailbedLayer::NailbedLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _start = pyDictGetInt(paramsDict, "start");
    _stride = pyDictGetInt(paramsDict, "stride");
    _outputsX = pyDictGetInt(paramsDict, "outputsX");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
}

void NailbedLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convBedOfNails(*_inputs[0], getActs(), _channels, _imgSize, _start, _stride, 0, 1);
}

void NailbedLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convBedOfNailsUndo(v, _prev[0]->getActsGrad(), _channels, _imgSize, _start, _stride, scaleTargets, 1);
}

/* 
 * =====================
 * GaussianBlurLayer
 * =====================
 */
GaussianBlurLayer::GaussianBlurLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _hFilter = pyDictGetMatrix(paramsDict, "filter");
}

void GaussianBlurLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convGaussianBlur(*_inputs[0], _filter, getActs(), true, _channels, 0, 1);
    convGaussianBlur(getActs(), _filter, getActs(), false, _channels, 0, 1);
}

// This is here just for completeness' sake. Why would you backpropagate
// through a blur filter?
void GaussianBlurLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& tgt1 = _prev[0]->getRcvdBInputs() > 0 ? _actGradsTmp : _prev[0]->getActsGrad();
    convGaussianBlur(v, _filter, tgt1, true, _channels, 0, 1);
    convGaussianBlur(tgt1, _filter, _prev[0]->getActsGrad(), false, _channels, scaleTargets, 1);
}

void GaussianBlurLayer::copyToGPU() {
    _filter.copyFromHost(*_hFilter, true);
}

/* 
 * =====================
 * ResizeLayer
 * =====================
 */
ResizeLayer::ResizeLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
    _tgtSize = pyDictGetInt(paramsDict, "tgtSize");
    _scale = pyDictGetFloat(paramsDict, "scale");
}

void ResizeLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResizeBilinear(*_inputs[0], getActs(), _imgSize, _tgtSize, _scale);
}

// Can't do this
void ResizeLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * RGBToYUVLayer
 * =====================
 */
RGBToYUVLayer::RGBToYUVLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
}

void RGBToYUVLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convRGBToYUV(*_inputs[0], getActs());
}

// Can't do this
void RGBToYUVLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * RGBToLABLayer
 * =====================
 */
RGBToLABLayer::RGBToLABLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _center = pyDictGetInt(paramsDict, "center");
}

void RGBToLABLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convRGBToLAB(*_inputs[0], getActs(), _center);
}

// Can't do this
void RGBToLABLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * ResponseNormLayer
 * =====================
 */
ResponseNormLayer::ResponseNormLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _size = pyDictGetInt(paramsDict, "size");

    _scale = pyDictGetFloat(paramsDict, "scale");
    _pow = pyDictGetFloat(paramsDict, "pow");
	_const = pyDictGetFloat(paramsDict, "rnorm_const");
}

void ResponseNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNorm(*_inputs[0], _denoms, getActs(), _channels, _size, _scale, _pow);
}

void ResponseNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormUndo(v, _denoms, _prev[0]->getActs(), getActs(), _prev[0]->getActsGrad(), _channels, _size, _scale, _pow, scaleTargets, 1);
}

void ResponseNormLayer::truncBwdActs() {
    Layer::truncBwdActs();
    if (_conserveMem) {
        _denoms.truncate();
    }
}

/* 
 * =====================
 * CrossMapResponseNormLayer
 * =====================
 */
CrossMapResponseNormLayer::CrossMapResponseNormLayer(ConvNet* convNet, PyObject* paramsDict) : ResponseNormLayer(convNet, paramsDict) {
    _blocked = pyDictGetInt(paramsDict, "blocked");
}

void CrossMapResponseNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormCrossMap(*_inputs[0], _denoms, getActs(), _channels, _size, _scale, _pow, _const, _blocked);
}

void CrossMapResponseNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormCrossMapUndo(v, _denoms, _prev[0]->getActs(), getActs(), _prev[0]->getActsGrad(), _channels, _size, _scale, _pow, _const,
								_blocked, scaleTargets, 1);
}

/* 
 * ===============================
 * CrossMapGlobalResponseNormLayer
 * ===============================
 */
CrossMapGlobalResponseNormLayer::CrossMapGlobalResponseNormLayer(ConvNet* convNet, PyObject* paramsDict) : 
ResponseNormLayer(convNet, paramsDict)
{
    _blocked = pyDictGetInt(paramsDict, "blocked");
}

void CrossMapGlobalResponseNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType)
{
#ifdef DEBUG
	if (0) //debug
	{
		cout << "CrossMapGlobalResponseNormLayer::fpropActs, " << _name << ", inpIdx=" << inpIdx << endl;
		_inputs[0]->print(2, 6);
	}
#endif
    convResponseNormCrossMap(*_inputs[0], _denoms, getActs(), _channels, 0, 0, _pow, 0, _blocked);
#ifdef DEBUG
	if (0) //debug
	{
		getActs().print(2, 6);
	}
#endif
}

void CrossMapGlobalResponseNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType)
{
    convResponseNormCrossMapUndo(v, _denoms, _prev[0]->getActs(), getActs(), _prev[0]->getActsGrad(), _channels, 0, 0,
								_pow, 0, _blocked, scaleTargets, 1);
#ifdef DEBUG
	if (0) //debug
	{
		cout << "CrossMapGlobalResponseNormLayer::bpropActs, " << _name << ", inpIdx=" << inpIdx << endl;
		_prev[0]->getActsGrad().print(2, 6);
	}
#endif
}



/* 
 * =====================
 * ContrastNormLayer
 * =====================
 */
ContrastNormLayer::ContrastNormLayer(ConvNet* convNet, PyObject* paramsDict) : ResponseNormLayer(convNet, paramsDict) {
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
}

void ContrastNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& images = *_inputs[0];
    convLocalPool(images, _meanDiffs, _channels, _size, -_size/2, 1, _imgSize, AvgPooler());
    _meanDiffs.add(images, -1, 1);
    convContrastNorm(images, _meanDiffs, _denoms, getActs(), _channels, _size, _scale, _pow);
}

void ContrastNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convContrastNormUndo(v, _denoms, _meanDiffs, getActs(), _prev[inpIdx]->getActsGrad(), _channels, _size, _scale, _pow, scaleTargets, 1);
}

void ContrastNormLayer::truncBwdActs() {
    ResponseNormLayer::truncBwdActs();
    if (_conserveMem) {
        _meanDiffs.truncate();
    }
}

/* 
 * =====================
 * CostLayer
 * =====================
 */
CostLayer::CostLayer(ConvNet* convNet, PyObject* paramsDict, bool trans) 
    : Layer(convNet, paramsDict, trans) {
    _coeff = pyDictGetFloat(paramsDict, "coeff");
}

float CostLayer::getCoeff() {
    return _coeff;
}

void CostLayer::bprop(PASS_TYPE passType) {
    if (_coeff != 0) {
        Layer::bprop(passType);
    }
}

bool CostLayer::isGradProducer() {
    return _coeff != 0;
}

doublev& CostLayer::getCost() {
    doublev& v = *new doublev();
    v.insert(v.begin(), _costv.begin(), _costv.end());
    return v;
}

CostLayer& CostLayer::makeCostLayer(ConvNet* convNet, string& type, PyObject* paramsDict) {
    if (type == "cost.logreg") {
        return *new LogregCostLayer(convNet, paramsDict);
    } else if (type == "cost.sum2") {
        return *new SumOfSquaresCostLayer(convNet, paramsDict);
    }
    throw string("Unknown cost layer type ") + type;
}

/* 
 * =====================
 * LogregCostLayer
 * =====================
 */
LogregCostLayer::LogregCostLayer(ConvNet* convNet, PyObject* paramsDict) : CostLayer(convNet, paramsDict, false) {
}

void LogregCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& probs = *_inputs[1];
        int numCases = labels.getNumElements();
        NVMatrix& trueLabelLogProbs = getActs(), correctProbs;
        computeLogregCost(labels, probs, trueLabelLogProbs, correctProbs);

	//	checkNaN(trueLabelLogProbs, "logreg output");
        _costv.clear();
        _costv.push_back(-trueLabelLogProbs.sum());
        _costv.push_back(numCases - correctProbs.sum());
    }

}

void LogregCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 1);
    NVMatrix& labels = _prev[0]->getActs();
    NVMatrix& probs = _prev[1]->getActs();
    NVMatrix& target = _prev[1]->getActsGrad();
    // Numerical stability optimization: if the layer below me is a softmax layer, let it handle
    // the entire gradient computation to avoid multiplying and dividing by a near-zero quantity.
    bool doWork = _prev[1]->getNext().size() > 1 || _prev[1]->getType() != "softmax";
    if (doWork) {
        computeLogregGrad(labels, probs, target, scaleTargets == 1, _coeff);
    }
}

/* 
 * =====================
 * SumOfSquaresCostLayer
 * =====================
 */
SumOfSquaresCostLayer::SumOfSquaresCostLayer(ConvNet* convNet, PyObject* paramsDict) : CostLayer(convNet, paramsDict, false) {
}

void SumOfSquaresCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _inputs[0]->apply(NVMatrixOps::Square(), getActs());
    _costv.clear();
    _costv.push_back(getActs().sum());
}

void SumOfSquaresCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _prev[inpIdx]->getActsGrad().add(*_inputs[0], scaleTargets, -2 * _coeff);
#ifdef DEBUG
	if (1) //debug
	{
		cout << "SumOfSquaresCostLayer::bpropActs, " << _name << endl;
		_prev[inpIdx]->getActsGrad().printShape("v");
		_prev[inpIdx]->getActsGrad().print(1, 50);
		checkNaN(_prev[inpIdx]->getActsGrad(), "prev acts grad");
	}
#endif
}
