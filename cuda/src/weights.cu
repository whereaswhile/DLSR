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

#include <weights.cuh>

bool Weights::_autoCopyToGPU = false;

void AdaDeltaWeights::update() {
        // Only true owner of weights updates
	
    if (_srcWeights == NULL) {
	assert(_onGPU);
	_gradAcc->applyBinary(NVMatrixBinaryOps::WeightedAddSquare(_rho, 1-_rho), getGrad()); 
	//printf("_grad size %d, %d, %d\n", getGrad().getNumRows(), getGrad().getNumCols(), getGrad().isTrans());
        //printf("_updateAcc size %d, %d, %d\n", _updateAcc->getNumRows(), _updateAcc->getNumCols(), _updateAcc->isTrans());

	getGrad().applyBinary(NVMatrixBinaryOps::AdaDeltaMult(_eps),  *_updateAcc); 	
	getGrad().applyBinary(NVMatrixBinaryOps::AdaDeltaDiv(_eps),  *_gradAcc);

	_updateAcc->applyBinary(NVMatrixBinaryOps::WeightedAddSquare(_rho, 1-_rho), getGrad());
	    
	_weights->add(getGrad(), 1);
	_numUpdates = 0;
    }
}
