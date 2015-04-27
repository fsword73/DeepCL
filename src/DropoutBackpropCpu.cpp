// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>
#include <cstring>

#include "OpenCLHelper.h"
#include "DropoutBackprop.h"
#include "StatefulTimer.h"

#include "DropoutBackpropCpu.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

DropoutBackpropCpu::DropoutBackpropCpu( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) :
        DropoutBackprop( cl, numPlanes, inputImageSize, dropRatio ) {
}
VIRTUAL void DropoutBackpropCpu::backward( int batchSize, uchar *mask,  float *gradOutput, float *gradInput ) {
    int totalLinearSize = batchSize * numPlanes * inputImageSize * inputImageSize;
    for( int i = 0; i < totalLinearSize; i++ ) {
        gradInput[i] = mask[i] == 1 ? gradOutput[i] : 0.0f;
    }
}
VIRTUAL void DropoutBackpropCpu::backward( int batchSize, CLWrapper *maskWrapper, CLWrapper *gradOutputWrapper, 
        CLWrapper *gradInputWrapper ) {
    StatefulTimer::instance()->timeCheck("DropoutBackpropCpu::backward start" );

    maskWrapper->copyToHost();
    gradOutputWrapper->copyToHost();

    uchar *mask = reinterpret_cast<uchar *>( maskWrapper->getHostArray() );
    float *gradOutput = reinterpret_cast<float *>( gradOutputWrapper->getHostArray() );
    float *gradInput = new float[ getInputSize( batchSize ) ];

    backward( batchSize, mask, gradOutput, gradInput );

    float *gradInputHostArray = reinterpret_cast<float *>( gradInputWrapper->getHostArray() );
    memcpy( gradInputHostArray, gradInput, sizeof(float) * getInputSize( batchSize ) );
    gradInputWrapper->copyToDevice();

    delete[] gradInput;
    
    StatefulTimer::instance()->timeCheck("DropoutBackpropCpu::backward end" );
}
