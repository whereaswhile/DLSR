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

#ifndef WEIGHTS_CUH
#define	WEIGHTS_CUH

#include <string>
#include <vector>
#include <iostream>
#include <helper_cuda.h>
#include <assert.h>
#include <nvmatrix.cuh>
#include <matrix.h>
#include "util.cuh"

using namespace std;

/*
 *  Virtual base class, keep the storage of actual weights & weigtsGrad
 *
 */
class Weights {
protected:
    Matrix* _hWeights;
    NVMatrix* _weights, *_weightsGrad;
    
    // _useGrad removed: 
    bool _onGPU;
    int _numUpdates;
    static bool _autoCopyToGPU;
    
    // Non-NULL if these weights are really shared from some other layer
    Weights* _srcWeights;

 
public:

    NVMatrix& getGrad() {
        assert(_onGPU);
        return *_weightsGrad;
    }

    NVMatrix& operator*() {
        return getW();
    }
    
    Weights(Weights& srcWeights) : _srcWeights(&srcWeights), _onGPU(false), _numUpdates(0),
                                               _weights(NULL), _weightsGrad(NULL){
        _hWeights = &srcWeights.getCPUW();
        //_useGrad = srcWeights.isUseGrad();   

    }

    Weights(Matrix& hWeights)
        : _srcWeights(NULL), _hWeights(&hWeights), _numUpdates(0), _onGPU(false), _weights(NULL),
           _weightsGrad(NULL) {
        if (_autoCopyToGPU) {
            copyToGPU();
        }
    }
        
    virtual ~Weights() {
        delete _hWeights;
        if (_srcWeights == NULL) {
            delete _weights;
            delete _weightsGrad;
        }
    }


    static void setAutoCopyToGPU(bool autoCopyToGPU) {
        _autoCopyToGPU = autoCopyToGPU;
    }
    
    NVMatrix& getW() {
        assert(_onGPU);
        return *_weights;
    }
    
        

    
    Matrix& getCPUW() {
        return *_hWeights;
    }
    
    
    int getNumRows() const {
        return _hWeights->getNumRows();
    }
    
    int getNumCols() const {
        return _hWeights->getNumCols();
    }
    
    virtual void copyToCPU() {
        if (_srcWeights == NULL) {
            assert(_onGPU);
            _weights->copyToHost(*_hWeights);
        }
    }
    
    // This function is assumed to be called in the order in which the layers
    // were defined
    virtual void copyToGPU() {
        if (_srcWeights == NULL) {
            _weights = new NVMatrix();
            _weights->copyFromHost(*_hWeights, true);
            _weightsGrad = new NVMatrix();
        } else {
            _weights = _srcWeights->_weights;
            _weightsGrad = _srcWeights->_weightsGrad;
        }
        _onGPU = true;
    }
    
    virtual void update() = 0;
    virtual bool isActive() = 0;

    // gradient is comming

    int incNumUpdates() {
        if (_srcWeights != NULL) {
            return _srcWeights->incNumUpdates();
        }
        return _numUpdates++;
    }
    
    // Returns the number of times a gradient has been computed for this
    // weight matrix during the current pass (interval between two calls of update())
    // through the net. This number will only be greater than 1 if this weight matrix
    // is *shared* by multiple layers in the net.
    int getNumUpdates() const {
        if (_srcWeights != NULL) {
            return _srcWeights->getNumUpdates();
        }
        return _numUpdates;
    }
    
};

class MomWeights: public Weights{
private:
    Matrix*  _hWeightsInc;
    NVMatrix*  _weightsInc;
    
    float _epsW, _wc, _mom;

    NVMatrix& getInc() {
        assert(_onGPU);
        return *_weightsInc;
    }
    Matrix& getCPUWInc() {
        return *_hWeightsInc;
    }
 
public:

    // shared
    MomWeights(MomWeights& srcWeights, float epsW) : Weights(srcWeights), _epsW(epsW), _wc(0), _weightsInc(NULL){
	
        _hWeightsInc = &srcWeights.getCPUWInc();
        _mom = srcWeights.getMom();
        if (_autoCopyToGPU) {
            copyToGPU();
        }
    }
    
    // unshared
    MomWeights(Matrix& hWeights, Matrix& hWeightsInc, float epsW, float wc, float mom)
        : Weights(hWeights), _hWeightsInc(&hWeightsInc), 
          _epsW(epsW), _wc(wc), _mom(mom), _weightsInc(NULL) {
        if (_autoCopyToGPU) {
            copyToGPU();
        }
    }
        
    ~MomWeights() {
        delete _hWeightsInc;
        if (_srcWeights == NULL) {
            delete _weightsInc;
        }
    }
  
 


    
    void copyToCPU() {
        if (_srcWeights == NULL) {
            assert(_onGPU);
            _weightsInc->copyToHost(*_hWeightsInc);
        }
        Weights::copyToCPU();
    }
    
    // This function is assumed to be called in the order in which the layers
    // were defined
    void copyToGPU() {
        if (_srcWeights == NULL) {
            _weightsInc = new NVMatrix();
            _weightsInc->copyFromHost(*_hWeightsInc, true);
        } else {
            _weightsInc = dynamic_cast<MomWeights*>(_srcWeights)->_weightsInc;
        }
        _onGPU = true;
        Weights::copyToGPU();
    }
    
    // Scale your gradient by epsW / numCases!
    void update() {
        // Only true owner of weights updates
        if (_srcWeights == NULL && _epsW > 0) {
            assert(_onGPU);
            _weightsInc->add(*_weightsGrad, _mom, _epsW);
            if (_wc > 0) {
                _weightsInc->add(*_weights, -_wc * _epsW);
            }
            _weights->add(*_weightsInc);
            _numUpdates = 0;
        }
    }
    
    bool isActive() {
        return _epsW > 0;
    }
    
    float getEps() const {
        return _epsW;
    }
    
    float getMom() const {
        return _mom;
    }
    
    float getWC() const {
        return _wc;
    }
};


class AdagradWeights: public Weights{
private:
    Matrix*  _hGradAcc;
    NVMatrix*  _gradAcc;
    
    float _eta;

    NVMatrix& getGradAcc() {
        assert(_onGPU);
        return *_gradAcc;
    }
    Matrix& getCPUGradAcc() {
        return *_hGradAcc;
    }
 
public:

    // shared
    AdagradWeights(AdagradWeights& srcWeights) : Weights(srcWeights),  _gradAcc(NULL){
	
        _hGradAcc = &srcWeights.getCPUGradAcc();
        _eta = srcWeights.getEta();
        if (_autoCopyToGPU) {
            copyToGPU();
        }
    }
    
    // unshared
    AdagradWeights(Matrix& hWeights, Matrix& hWeightsGradAcc, float etaW)
        : Weights(hWeights), _hGradAcc(&hWeightsGradAcc), _eta(etaW), _gradAcc(NULL) {
        if (_autoCopyToGPU) {
            copyToGPU();
        }
    }
        
    ~AdagradWeights() {
        delete _hGradAcc;
        if (_srcWeights == NULL) {
            delete _gradAcc;
        }
    }
  
 


    
    void copyToCPU() {
        if (_srcWeights == NULL) {
            assert(_onGPU);
            _gradAcc->copyToHost(*_hGradAcc);
        }
        Weights::copyToCPU();
    }
    
    // This function is assumed to be called in the order in which the layers
    // were defined
    void copyToGPU() {
        if (_srcWeights == NULL) {
            _gradAcc = new NVMatrix();
            _gradAcc->copyFromHost(*_hGradAcc, true);
        } else {
            _gradAcc = dynamic_cast<AdagradWeights*>(_srcWeights)->_gradAcc;
        }
        _onGPU = true;
        Weights::copyToGPU();
    }
    
    // acc += grad^2 
    // weights -= eta * grad / (fudge_factor + sqrt(acc))
    void update() {
        // Only true owner of weights updates
	const float fudge_factor = 1e-6;
        if (_srcWeights == NULL) {
            assert(_onGPU);
	    _gradAcc->applyBinary(NVMatrixBinaryOps::AddSquare(), getGrad()); 
	    getGrad().applyBinary(NVMatrixBinaryOps::AdaGrad(fudge_factor, _eta),  *_gradAcc); 
            _weights->add(getGrad());
            _numUpdates = 0;
        }
    }
    
    bool isActive() {
        return true;
    }
    
    float getEta() const {
        return _eta;
    }
    

};


class AdaDeltaWeights: public Weights{
private:
    Matrix*  _hGradAcc;
    Matrix*  _hUpdateAcc;
    NVMatrix*  _gradAcc, *_updateAcc;
    
    float _eps, _rho;

    NVMatrix& getGradAcc() {
        assert(_onGPU);
        return *_gradAcc;
    }
    Matrix& getCPUGradAcc() {
        return *_hGradAcc;
    }

    NVMatrix& getUpdateAcc() {
        assert(_onGPU);
        return *_updateAcc;
    }
    Matrix& getCPUUpdateAcc() {
        return *_hUpdateAcc;
    }
 
public:

    // shared
    AdaDeltaWeights(AdaDeltaWeights& srcWeights) : Weights(srcWeights),  _gradAcc(NULL), _updateAcc(NULL){
	
        _hGradAcc = &srcWeights.getCPUGradAcc();
	_hUpdateAcc = &srcWeights.getCPUUpdateAcc();
        _eps = srcWeights.getEps();
	_rho = srcWeights.getRho();
        if (_autoCopyToGPU) {
            copyToGPU();
        }
    }
    
    // unshared
    AdaDeltaWeights(Matrix& hWeights, Matrix& hWeightsGradAcc, Matrix& hWeightsUpdateAcc, float epsW, float rhoW)
        : Weights(hWeights), _hGradAcc(&hWeightsGradAcc), _hUpdateAcc(&hWeightsUpdateAcc), _eps(epsW), _rho(rhoW), _gradAcc(NULL), _updateAcc(NULL) {
        if (_autoCopyToGPU) {
            copyToGPU();
        }
    }
        
    ~AdaDeltaWeights() {
        delete _hGradAcc;
        delete _hUpdateAcc;
        if (_srcWeights == NULL) {
            delete _gradAcc;
            delete _updateAcc;
        }
    }
  
 


    
    void copyToCPU() {
        if (_srcWeights == NULL) {
            assert(_onGPU);
            _gradAcc->copyToHost(*_hGradAcc);
            _updateAcc->copyToHost(*_hUpdateAcc);
        }
        Weights::copyToCPU();
    }
    
    // This function is assumed to be called in the order in which the layers
    // were defined
    void copyToGPU() {
        if (_srcWeights == NULL) {
            _gradAcc = new NVMatrix();
            _gradAcc->copyFromHost(*_hGradAcc, true);
            _updateAcc = new NVMatrix();
            _updateAcc->copyFromHost(*_hUpdateAcc, true);

        } else {
            _gradAcc = dynamic_cast<AdaDeltaWeights*>(_srcWeights)->_gradAcc;
            _updateAcc = dynamic_cast<AdaDeltaWeights*>(_srcWeights)->_updateAcc;
        }
        _onGPU = true;
        Weights::copyToGPU();
    }
    
    // gradacc = rho*gradacc + (1-rho)grad^2
    // RMS_gradacc = sqrt(gradacc + eps)
    // RMS_updateacc = sqrt(updateacc + eps)
    // update = RMS(updateAcc)/RMS(gradAcc) * grad   ???
    // updateAcc = rho*updateAcc + (1-rho) update^2
    // weights += update
     
    // weights -= eta * grad / (fudge_factor + sqrt(acc))
    void update();
    bool isActive() {
        return true;
    }
    
    float getEps() const {
        return _eps;
    }

    float getRho() const {
	return _rho;
    }
    

};

class WeightList {
private:
    std::vector<Weights*> _weightList;

public:
    Weights& operator[](const int idx) const {
        return *_weightList[idx];
    }
    
    ~WeightList() {
        for (int i = 0; i < _weightList.size(); i++) {
            delete _weightList[i];
        }
    }
    
//    WeightList(MatrixV& hWeights, MatrixV& hWeightsInc, floatv& epsW, floatv& wc, floatv& mom, bool useGrads) : _initialized(false) {
//        initialize(hWeights, hWeightsInc, epsW, wc, mom, useGrads);
//    }
    
    WeightList() {
    }
    
//    void initialize(MatrixV& hWeights, MatrixV& hWeightsInc, floatv& epsW, floatv& wc, floatv& mom, bool useGrads) {
//        for (int i = 0; i < hWeights.size(); i++) {
//            _weightList.push_back(new Weights(*hWeights[i], *hWeightsInc[i], epsW[i], wc[i], mom[i], useGrads));
//        }
//        _initialized = true;
//        delete &hWeights;
//        delete &hWeightsInc;
//        delete &epsW;
//        delete &wc;
//        delete &mom;
//    }
    
    void addWeights(Weights& w) {
        _weightList.push_back(&w);
    }
    
//    void addWeights(WeightList& wl) {
//        for (int i = 0; i < wl.getSize(); i++) {
//            addWeights(wl[i]);
//        }
//    }
    
    void update() {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->update();
        }
    }

    void copyToCPU() {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->copyToCPU();
        }
    }

    void copyToGPU() {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->copyToGPU();
        }
    }
    
    int getSize() {
        return _weightList.size();
    }
};

#endif	/* WEIGHTS_CUH */
