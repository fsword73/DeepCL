// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "conv/ForwardTiled.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "conv/AddBias.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL ForwardTiled::~ForwardTiled() {
    delete kernel;
    delete addBias;
}
VIRTUAL void ForwardTiled::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper) {
    StatefulTimer::timeCheck("ForwardTiled::forward START");

    kernel->in(batchSize);
    kernel->input(dataWrapper);
    kernel->input(weightsWrapper);
    kernel->output(outputWrapper);

	//ToDo
	int workgroupsize =64;
	const int WAVE_SIZE = 64; 	//max of AMD: 64 and NV:  32 
	int workgroupPerImage =1;
	if (dim.outputSize > 19)
	{		
		workgroupsize = 256;
		if (dim.filterSize == 3 && dim.outputSize > 32)
		{
			//16x64 tile
			const int HSIZE = 16;
			const int VSIZE = 64;
			workgroupPerImage = ((dim.outputSize + HSIZE - 1) / HSIZE) * ((dim.outputSize + VSIZE - 1) / VSIZE);
		}
		else
		{ 
			//32x32 tile
			const int HSIZE = 32;
			const int VSIZE = 32;
			workgroupPerImage = ((dim.outputSize + HSIZE - 1) / HSIZE) * ((dim.outputSize + VSIZE - 1) / VSIZE);
		}
	}
	else
	{
		int workgroupsize = (dim.outputSizeSquared / 2 + WAVE_SIZE - 1) & (!(WAVE_SIZE - 1));
		if (workgroupsize < 64)
			workgroupsize = 64;
		workgroupPerImage = 1;
	}


    int globalSize = batchSize * dim.numFilters * workgroupPerImage * workgroupsize;    
 	//    cout << "ForwardTiled globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;

    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();
    StatefulTimer::timeCheck("ForwardTiled::forward after call forward");

    if(dim.biased) {
        addBias->forward(
            batchSize, dim.numFilters, dim.outputSize,
            outputWrapper, biasWrapper);
    }
    StatefulTimer::timeCheck("ForwardTiled::forward END");
}
ForwardTiled::ForwardTiled(EasyCL *cl, LayerDimensions dim) :
            Forward(cl, dim)
        {
    addBias = new AddBias(cl);

    std::string options = "";
    options += dim.buildOptionsString();

	//ToDo
	if (dim.outputSize > 19)
	{
		if(dim.filterSize == 3 && dim.outputSize > 32)
		{   //16x64 tile
			options += " -D TILE_WIDTH=16";
			options += " -D TILE_HEIGHT=16";
			options += " -D VTILE_REPEAT=4";
			options += " -D FIXED_WORKGROUP_SIZE=256";
		}
		else
		{   //32x32 tile
			options += " -D TILE_WIDTH=32";
			options += " -D TILE_HEIGHT=8";
			options += " -D VTILE_REPEAT=4";
			options += " -D FIXED_WORKGROUP_SIZE=256";
		}

	}
	else
	{
		const int WAVE_SIZE = 64; 
		//max of AMD: 64 and NV:  32 
		int FIXED_WORKGROUP_SIZE = (dim.outputSizeSquared /2 + WAVE_SIZE - 1) & (!(WAVE_SIZE - 1));
		if (FIXED_WORKGROUP_SIZE < 64)
			FIXED_WORKGROUP_SIZE = 64;
		int VTILE_HEGIHT = FIXED_WORKGROUP_SIZE / dim.outputSize;
		int VTILE_REPEAT = (dim.outputSizeSquared + FIXED_WORKGROUP_SIZE - 1) / FIXED_WORKGROUP_SIZE;

		options += " -D TILE_WIDTH=" + toString(dim.outputSize);
		options += " -D TILE_HEIGHT=" + toString(VTILE_HEGIHT);
		options += " -D VTILE_REPEAT=" + toString(VTILE_REPEAT);
		options += " -D FIXED_WORKGROUP_SIZE=" + toString(FIXED_WORKGROUP_SIZE);
	}

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/forwardTiled.cl", "convolve_tilemode_float", 'options')
    // ]]]
    // generated using cog, from cl/forwardTiled.cl:
    const char * kernelSource =  
    "// images are organized like [imageId][plane][row][col]\n"
    "// filters are organized like [filterid][inplane][filterrow][filtercol]\n"
    "// output are organized like [imageid][filterid][row][col]\n"
    "// global group id is organized like output, ie: [imageid][outplane][HTILE_ID][VTILE_ID]\n"
    "\n"
    "\n"
    "#if 0 //REAL ARGS\n"
    "#define BIASED\n"
    "#define gNumInputPlanes 1\n"
    "#define gInputPlanes 1\n"
    "#define gInputSize 28\n"
    "#define gInputSizeSquared 784\n"
    "#define gNumFilters 8\n"
    "#define gFilterSize 5\n"
    "#define gHalfFilterSize 2\n"
    "#define gFilterSizeSquared 25\n"
    "#define gNumOutputPlanes 8\n"
    "#define gOutputPlanes 150\n"
    "#define gOutputSize 28\n"
    "#define gOutputSizeSquared (28*28)\n"
    "#define gPadZeros 1\n"
    "#define gMargin 2\n"
    "#define gEven 0\n"
    "#define gSkip 0\n"
    "#define TILE_WIDTH 32\n"
    "#define TILE_HEIGHT 8\n"
    "#define VTILE_REPEAT 4\n"
    "#define FIXED_WORKGROUP_SIZE 256\n"
    "#endif\n"
    "#if 0\n"
    "#define gOutputSize 64\n"
    "#define gOutputSizeSquared ( gOutputSize * gOutputSize)\n"
    "#define gNumFilters 8\n"
    "#define gNumInputPlanes 8\n"
    "#define gInputSize 64\n"
    "#define gInputSizeSquared (gInputSize *gInputSize)\n"
    "#define gFilterSize  3\n"
    "#define gHalfFilterSize  ( gFilterSize >>1)\n"
    "#define gFilterSizeSquared (gFilterSize *gFilterSize )\n"
    "#define gEven 0\n"
    "\n"
    "//Following 4 lines to be adjusted with differnt gOutputSize & gFilterSize\n"
    "#define TILE_WIDTH   16\n"
    "#define TILE_HEIGHT  16\n"
    "\n"
    "//Calculate 4 Pixels per thread o reuse S_Load_dwordx4\n"
    "#define VTILE_REPEAT 4\n"
    "#define FIXED_WORKGROUP_SIZE  256\n"
    "#endif\n"
    "//The bottleneck of this kernel is : VALU, Scarlar, LDS,  not buffer loading\n"
    "\n"
    "//go: 19x19 : TILE_WIDTH = 19, TILE_HEIGHT=2 ,  TILE_REPEAT: 2\n"
    "//default: FilterSize; 3x3  : TILE_WIDTH = 16, TILE_HEIGHT = 16  TILE_REPEAT: 2\n"
    "//Other  : FilterSize; 5x5  : TILE_WIDTH = 32, TILE_HEIGHT = 8   TILE_REPEAT: 4\n"
    "//32x32     :        				: TILE_WIDTH = 32, TILE_HEIGHT = 8   TILE_REPEAT: 4\n"
    "\n"
    "\n"
    "#define HTILES ((gOutputSize + TILE_WIDTH-1) / TILE_WIDTH )\n"
    "#define VTILES ((gOutputSize + TILE_HEIGHT-1) / (TILE_HEIGHT * VTILE_REPEAT))\n"
    "#define HTILE_LOCAL_SIZE ( gFilterSize -1 + TILE_WIDTH)\n"
    "#define VTILE_LOCAL_SIZE ( gFilterSize -1 + TILE_HEIGHT * VTILE_REPEAT)\n"
    "\n"
    "\n"
    "#define ROWS_PER_WORKGROUP    ( FIXED_WORKGROUP_SIZE / HTILE_LOCAL_SIZE )\n"
    "#define MAX_VALID_ID          ( (FIXED_WORKGROUP_SIZE /TILE_WIDTH ) * TILE_WIDTH)\n"
    "\n"
    "// each time  it fetches   rows == ROWS_PER_WORKGROUP .\n"
    "#define IMAGE_LOAD_ITERATIONS ( (HTILE_LOCAL_SIZE + ROWS_PER_WORKGROUP-1) / ROWS_PER_WORKGROUP )\n"
    "\n"
    "//Load image  into LDS once\n"
    "//Load by Rows\n"
    "//Load Filters by SGPR  once\n"
    "\n"
    "#if gOutputSize == 1\n"
    "void convolve_1x1_float(\n"
    "    const int batchSize,\n"
    "    global const float *inputs, global const float *filters,\n"
    "    global float *output, __local float* sdata, int globalId, int localId)\n"
    "{\n"
    "    int outputImage2Id = globalId / gOutputSizeSquared;\n"
    "    int exampleId = outputImage2Id / gNumFilters;\n"
    "    int filterId = outputImage2Id % gNumFilters;\n"
    "\n"
    "    // intraimage coords\n"
    "    int localid = globalId % gOutputSizeSquared;\n"
    "    int outputRow = localid / gOutputSize;\n"
    "    int outputCol = localid % gOutputSize;\n"
    "\n"
    "    global float const*inputCube = inputs + exampleId * gNumInputPlanes * gInputSizeSquared;\n"
    "    global float const*filterCube = filters + filterId * gNumInputPlanes * gFilterSizeSquared;\n"
    "\n"
    "    float sum = 0;\n"
    "    if (exampleId < batchSize) {\n"
    "#define iterations (( gNumInputPlanes + FIXED_WORKGROUP_SIZE -1 ) / FIXED_WORKGROUP_SIZE)\n"
    "				for(int i= 0; i < iterations; i++)\n"
    "				{\n"
    "						int inputPlaneIdx = localId + i* FIXED_WORKGROUP_SIZE;\n"
    "						if(inputPlaneIdx >= gNumInputPlanes)\n"
    "							   break;\n"
    "\n"
    "            global float const*inputPlane = inputCube + inputPlaneIdx * gInputSizeSquared;\n"
    "            global float const*filterPlane = filterCube + inputPlaneIdx * gFilterSizeSquared;\n"
    "            for (int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++) {\n"
    "                // trying to reduce register pressure...\n"
    "                #if gPadZeros == 1\n"
    "                    #define inputRowIdx (outputRow + u)\n"
    "                #else\n"
    "                    #define inputRowIdx (outputRow + u + gHalfFilterSize)\n"
    "                #endif\n"
    "                global float const *inputRow = inputPlane + inputRowIdx * gInputSize;\n"
    "                global float const *filterRow = filterPlane + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n"
    "                bool rowOk = inputRowIdx >= 0 && inputRowIdx < gInputSize;\n"
    "                #pragma unroll\n"
    "                for (int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++) {\n"
    "                    #if gPadZeros == 1\n"
    "                        #define inputColIdx (outputCol + v)\n"
    "                    #else\n"
    "                        #define inputColIdx (outputCol + v + gHalfFilterSize)\n"
    "                    #endif\n"
    "                    bool process = rowOk && inputColIdx >= 0 && inputColIdx < gInputSize;\n"
    "                    if (process) {\n"
    "                            sum += inputRow[inputColIdx] * filterRow[v];\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "\n"
    "		//Reduction\n"
    "\n"
    "		//store into local\n"
    "		sdata[localId] = sum;\n"
    "		barrier(CLK_LOCAL_MEM_FENCE);\n"
    "		for(unsigned int s = FIXED_WORKGROUP_SIZE >>1; s > 0; s >>= 1)\n"
    "		{\n"
    "			if(localId < s)\n"
    "			{\n"
    "				sdata[localId] += sdata[localId + s];\n"
    "			}\n"
    "			barrier(CLK_LOCAL_MEM_FENCE);\n"
    "		}\n"
    "\n"
    "		if(localId == 0)\n"
    "		{\n"
    "			output[globalId] = sdata[0];\n"
    "		}\n"
    "}\n"
    "#endif\n"
    "\n"
    "#if 0\n"
    "void kernel test_kernel(__global const float* filter,\n"
    "						__global const float* inputs,\n"
    "						__global const float* filters,\n"
    "						__global const float* weights,\n"
    "						__global const float* dataBuf4,\n"
    "						__global const float* dataBuf5,\n"
    "						__global float* output,\n"
    "						const int batchSize,\n"
    "						const int const2,\n"
    "						const int const3,\n"
    "						const int const4,\n"
    "						const int const5,\n"
    "						const int const6)\n"
    "#else\n"
    "void kernel convolve_tilemode_float(\n"
    "    const int batchSize,\n"
    "    global const float *inputs, global const float *filters,\n"
    "    global float *output)\n"
    "#endif\n"
    "{\n"
    "\n"
    "		int localId = get_local_id(0);\n"
    "		int groupId = get_group_id(0);\n"
    "\n"
    "#if gOutputSize == 1\n"
    "		__local float sdata[FIXED_WORKGROUP_SIZE];\n"
    "    convolve_1x1_float(batchSize, inputs, filters, output, sdata, groupId, localId);\n"
    "#else\n"
    "		int outputImage2Id = 	groupId /(HTILES * VTILES);\n"
    "    int batchId  = outputImage2Id / gNumFilters;\n"
    "    int filterId = outputImage2Id % gNumFilters;\n"
    "		float sum = 0;\n"
    "	  float sum2 = 0;\n"
    "		float sum3 = 0;\n"
    "	  float sum4 = 0;\n"
    "		__local float __inputs[HTILE_LOCAL_SIZE * VTILE_LOCAL_SIZE];\n"
    "\n"
    "	  int filterIdOffset = filterId * gNumInputPlanes * gFilterSize * gFilterSize;\n"
    "\n"
    "\n"
    "		//Step1: calculuating output\n"
    "		int tile_BL_X, tile_BL_Y;\n"
    "		tile_BL_Y = ((groupId % (HTILES * VTILES)) / HTILES) * TILE_WIDTH;\n"
    "		tile_BL_X = ((groupId % (HTILES * VTILES)) % HTILES) * TILE_WIDTH;\n"
    "\n"
    "		//Offset in Local Memory\n"
    "		int localRow  = localId / TILE_WIDTH ;\n"
    "		int localCol  = localId % TILE_WIDTH ;\n"
    "\n"
    "\n"
    "		//Offset in current Memory\n"
    "		int outputRow = localId / TILE_WIDTH  + tile_BL_Y;\n"
    "		int outputCol	= localId % TILE_WIDTH  + tile_BL_X;\n"
    "\n"
    "		bool bValid =  outputRow < gOutputSize && outputCol < gOutputSize && localId < MAX_VALID_ID;\n"
    "\n"
    "		//2nd pixel vaid or not\n"
    "		bool bValid2 =  (outputRow + TILE_HEIGHT)   < gOutputSize && outputCol < gOutputSize && localId < MAX_VALID_ID;\n"
    "		bool bValid3 =  (outputRow + TILE_HEIGHT*2) < gOutputSize && outputCol < gOutputSize && localId < MAX_VALID_ID;\n"
    "		bool bValid4 =  (outputRow + TILE_HEIGHT*3) < gOutputSize && outputCol < gOutputSize && localId < MAX_VALID_ID;\n"
    "\n"
    "		//[Batch][InputPlane][Row][col]\n"
    "		int inputImageOffset = batchId * gNumInputPlanes * gInputSizeSquared ;\n"
    "\n"
    "		//Step2: loading_offset;\n"
    "    int local_x, local_y;\n"
    "		bool bImageLoad 	= false;\n"
    "		bool bValidX = false;\n"
    "		int image_x, image_y;\n"
    "\n"
    "		//only partial threads to load Input\n"
    "		if( localId < (ROWS_PER_WORKGROUP * HTILE_LOCAL_SIZE))\n"
    "		{\n"
    "			  bImageLoad = true;\n"
    "				//Local memory Offset\n"
    "				local_x	= localId % HTILE_LOCAL_SIZE;\n"
    "			  local_y = localId / HTILE_LOCAL_SIZE;\n"
    "\n"
    "			  //load Image Offset\n"
    "			  image_x = local_x + tile_BL_X;\n"
    "			  image_y = local_y + tile_BL_Y;\n"
    "\n"
    "#if gPadZeros == 1\n"
    "			  image_x -= gHalfFilterSize;\n"
    "			  image_y -= gHalfFilterSize;\n"
    "#endif\n"
    "\n"
    "\n"
    "				//reduce operation for inner loop\n"
    "			  bValidX = image_x >=0 && image_x < gInputSize;\n"
    "		}\n"
    "\n"
    "\n"
    "\n"
    "		//step3: Fordward\n"
    "		int forceloop = min(gNumInputPlanes, gNumInputPlanes*batchSize);\n"
    "		for (int inputPlaneIdx = 0; inputPlaneIdx < forceloop; inputPlaneIdx++) {\n"
    "\n"
    "			  //only bImageLoad == true will load images\n"
    "			  if(bImageLoad)\n"
    "				{\n"
    "						for(int i = 0; i < IMAGE_LOAD_ITERATIONS; i++)\n"
    "						{\n"
    "								int localOffset =	(local_y + i * ROWS_PER_WORKGROUP)  * HTILE_LOCAL_SIZE + local_x;\n"
    "\n"
    "								//boundary check\n"
    "								if( localOffset < (HTILE_LOCAL_SIZE * VTILE_LOCAL_SIZE))\n"
    "								{\n"
    "										float value;\n"
    "										int row = 0;\n"
    "										bool bProcess = false;\n"
    "\n"
    "										row = image_y + i * ROWS_PER_WORKGROUP;\n"
    "										bProcess = row >= 0 && row < gInputSize && bValidX;\n"
    "										unsigned inputOffset =  inputImageOffset +\n"
    "																		 inputPlaneIdx * gInputSizeSquared +\n"
    "																		 row * gInputSize +\n"
    "																		 image_x;\n"
    "									 value =  inputs[inputOffset];\n"
    "										if( !bProcess)\n"
    "										{\n"
    "											value = 0;\n"
    "										}\n"
    "\n"
    "										__inputs[localOffset] = value;\n"
    "								}\n"
    "						}\n"
    "				}\n"
    "\n"
    "				barrier(CLK_LOCAL_MEM_FENCE);\n"
    "\n"
    "				// SUM +=\n"
    "#if (gFilterSize>5)\n"
    "				//minimized loading of constant for big gFilterSize\n"
    "				if(bValid)\n"
    "				{\n"
    "						  //Fix the LLVM performance bug for address calculuating\n"
    "						  //int filterIdx  = 	filterIdOffset + inputPlaneIdx * gFilterSizeSquared +  i * gFilterSize + j;\n"
    "							__global const *f = &filters[filterIdOffset + inputPlaneIdx * gFilterSizeSquared];\n"
    "\n"
    "							for(int i = 0; i < gFilterSize; i++)\n"
    "							{\n"
    "									for(int j =0; j < gFilterSize; j++)\n"
    "									{\n"
    "											//image COL 0, 1 ,2 * filter COL 0, 1, 2\n"
    "											unsigned int inputIdx   = (localRow +  i) * HTILE_LOCAL_SIZE +\n"
    "																								 (localCol  + j);\n"
    "										  float thisWeight = *(f + i * gFilterSize + j);\n"
    "											sum += __inputs[inputIdx] * thisWeight;\n"
    "									}\n"
    "									if(bValid2)\n"
    "									for(int j =0; j < gFilterSize; j++)\n"
    "									{\n"
    "\n"
    "											unsigned int inputIdx   = (localRow +  i + TILE_HEIGHT) * HTILE_LOCAL_SIZE +\n"
    "																								(localCol + j);\n"
    "\n"
    "											float thisWeight = *(f + i * gFilterSize + j);\n"
    "											sum2 += __inputs[inputIdx] * thisWeight;\n"
    "									}\n"
    "									if(bValid3)\n"
    "									for(int j =0; j < gFilterSize; j++)\n"
    "									{\n"
    "\n"
    "											unsigned int inputIdx   = (localRow  + i + TILE_HEIGHT*2) * HTILE_LOCAL_SIZE +\n"
    "																			 (localCol  + j);\n"
    "\n"
    "										float thisWeight = *(f + i * gFilterSize + j);\n"
    "										sum3 += __inputs[inputIdx] * thisWeight;\n"
    "									}\n"
    "\n"
    "									if(bValid4)\n"
    "									for(int j =0; j < gFilterSize; j++)\n"
    "									{\n"
    "											unsigned int inputIdx   = (localRow  + i + TILE_HEIGHT*3) * HTILE_LOCAL_SIZE +\n"
    "																			 (localCol  + j);\n"
    "											float thisWeight = *(f + i * gFilterSize + j);\n"
    "											sum4 += __inputs[inputIdx] * thisWeight;\n"
    "									}\n"
    "						}\n"
    "			 }\n"
    "#else\n"
    "				//Minimized Scalar Instructions\n"
    "				if(bValid)\n"
    "				{\n"
    "							for( int i = 0; i < gFilterSize; i++)\n"
    "							{\n"
    "								for(int j =0; j < gFilterSize; j++)\n"
    "								{\n"
    "										//image COL 0, 1 ,2 * filter COL 0, 1, 2\n"
    "										int inputIdx   = (localRow  + i) * HTILE_LOCAL_SIZE +\n"
    "																		 (localCol  + j);\n"
    "\n"
    "									  int filterIdx  = filterIdOffset +\n"
    "																				inputPlaneIdx * gFilterSizeSquared +\n"
    "																			i * gFilterSize +\n"
    "																			j;\n"
    "\n"
    "\n"
    "\n"
    "										sum += __inputs[inputIdx] * filters[filterIdx];\n"
    "								}\n"
    "							}\n"
    "								if(bValid2){\n"
    "									for( int i = 0; i < gFilterSize; i++)\n"
    "										for(int j =0; j < gFilterSize; j++)\n"
    "										{\n"
    "												int inputIdx   =  (localRow + i +TILE_HEIGHT) * HTILE_LOCAL_SIZE +\n"
    "																					(localCol + j);\n"
    "\n"
    "												int filterIdx  = 	filterIdOffset +\n"
    "																					inputPlaneIdx * gFilterSizeSquared +\n"
    "																					i * gFilterSize +\n"
    "																					j;\n"
    "\n"
    "												sum2 += __inputs[inputIdx] * filters[filterIdx];\n"
    "										}\n"
    "								}\n"
    "								if(bValid3){\n"
    "									for( int i = 0; i < gFilterSize; i++)\n"
    "										for(int j =0; j < gFilterSize; j++)\n"
    "										{\n"
    "												//shift gHalfFilterSize\n"
    "												int inputIdx   =  (localRow + i +TILE_HEIGHT*2) * HTILE_LOCAL_SIZE +\n"
    "																					(localCol + j);\n"
    "\n"
    "												int filterIdx  = 	filterIdOffset +\n"
    "																					inputPlaneIdx * gFilterSizeSquared +\n"
    "																					i * gFilterSize +\n"
    "																					j;\n"
    "\n"
    "												sum3 += __inputs[inputIdx] * filters[filterIdx];\n"
    "										}\n"
    "								}\n"
    "								if(bValid4){\n"
    "									for( int i = 0; i < gFilterSize; i++)\n"
    "										for(int j =0; j < gFilterSize; j++)\n"
    "										{\n"
    "												//shift gHalfFilterSize\n"
    "												int inputIdx   =  (localRow + i +TILE_HEIGHT*3) * HTILE_LOCAL_SIZE +\n"
    "																					(localCol + j);\n"
    "\n"
    "												int filterIdx  = 	filterIdOffset +\n"
    "																					inputPlaneIdx * gFilterSizeSquared +\n"
    "																					i * gFilterSize +\n"
    "																					j;\n"
    "\n"
    "												sum4 += __inputs[inputIdx] * filters[filterIdx];\n"
    "										}\n"
    "								}\n"
    "\n"
    "				}\n"
    "#endif\n"
    "		}\n"
    "\n"
    "\n"
    "		if (bValid)\n"
    "		{\n"
    "\n"
    "		   unsigned int outputOffset  = outputImage2Id * gOutputSizeSquared + outputRow * gOutputSize + outputCol;\n"
    "\n"
    "       output[outputOffset] = sum;\n"
    "			 if(bValid2)\n"
    "			 {\n"
    "				 output[outputOffset + TILE_HEIGHT * gOutputSize] = sum2;\n"
    "			 }\n"
    "			 if(bValid3)\n"
    "			 {\n"
    "				 output[outputOffset + TILE_HEIGHT * 2 * gOutputSize] = sum3;\n"
    "			 }\n"
    "			 if(bValid4)\n"
    "			 {\n"
    "				 output[outputOffset + TILE_HEIGHT * 3 * gOutputSize] = sum4;\n"
    "			 }\n"
    "\n"
    "    }\n"
    "#endif\n"
    "}\n"
    "\n"
    "\n"
    "\n"
    "\n"
    "#if 0\n"
    "VIRTUAL float *ForwardCpu::forward(int batchSize, float *inputData, float *weights, float *bias) {\n"
    "//    cout << \"ForwardCpu::forward outputcubesize=\" << dim.outputCubeSize << \" batchSize=\" << batchSize << endl;\n"
    "    float *output = new float[ dim.outputCubeSize * batchSize ];\n"
    "    for(int n = 0; n < batchSize; n++) {\n"
    "        for(int filter = 0; filter < dim.numFilters; filter++) {\n"
    "            for(int outRow = 0; outRow < dim.outputSize; outRow += 1 + dim.skip) {\n"
    "                for(int outCol = 0; outCol < dim.outputSize; outCol += 1 + dim.skip) {\n"
    "                    float sum = 0;\n"
    "                    for(int inPlane = 0; inPlane < dim.inputPlanes; inPlane++) {\n"
    "//                        cout << \"inplane=\" << inPlane << endl;\n"
    "                        for(int u = -dim.halfFilterSize; u <= dim.halfFilterSize; u++) {\n"
    "                            int inRow = outRow * (dim.skip + 1) + u + (dim.padZeros ? 0 : dim.halfFilterSize);\n"
    "//                                cout << \"candidate inRow \" << inRow << endl;\n"
    "                            if(inRow < 0 || inRow > dim.inputSize - 1) {\n"
    "                                continue;\n"
    "                            }\n"
    "                            int filterRow = u + dim.halfFilterSize;\n"
    "                            for(int v = -dim.halfFilterSize; v <= dim.halfFilterSize; v++) {\n"
    "                                int inCol = outCol * (dim.skip + 1) + v + (dim.padZeros ? 0 : dim.halfFilterSize);\n"
    "                                int filterCol = v + dim.halfFilterSize;\n"
    "                                if(inCol < 0 || inCol > dim.inputSize - 1) {\n"
    "                                    continue;\n"
    "                                }\n"
    "                                int inputIndex = (( n\n"
    "                                    * dim.inputPlanes + inPlane)\n"
    "                                    * dim.inputSize + inRow)\n"
    "                                    * dim.inputSize + inCol;\n"
    "                                int weightIndex = (( filter\n"
    "                                    * dim.inputPlanes + inPlane)\n"
    "                                    * dim.filterSize  + filterRow)\n"
    "                                    * dim.filterSize  + filterCol;\n"
    "//                                    cout << \"inpos \" << inRow << \",\" << inCol << \" outpos \" << outRow << \",\" << outCol\n"
    "//                                        << \" filterpos \" << filterRow << \",\" << filterCol << endl;\n"
    "                                float sumchange = inputData[ inputIndex] * weights[ weightIndex ];\n"
    "                                if(sumchange != 0) {\n"
    "//                                        cout << inputData[inputIndex] << \" * \" << weights[weightIndex] << \" = \" << sumchange << endl;\n"
    "                                }\n"
    "                                sum += sumchange;\n"
    "//                                cout << \"inputIndex=\" << inputIndex << \" weightIndex=\" << weightIndex <<\n"
    "//                                    \"  inputData[inputIndex]=\" << inputData[inputIndex] << \" weights[weightIndex]=\" << weights[weightIndex] << \" sumchange \" << sumchange << \" sum=\" << sum << endl;\n"
    "                            }\n"
    "                        }\n"
    "                    }\n"
    "                    if(dim.biased) {\n"
    "                        sum += bias[filter];\n"
    "                    }\n"
    "//                    sum = fn->calc(sum);\n"
    "                    int outputIndex = (( n\n"
    "                        * dim.numFilters + filter)\n"
    "                        * dim.outputSize + outRow)\n"
    "                        * dim.outputSize + outCol;\n"
    "                    output[outputIndex] = sum;\n"
    "//                    cout << \"outputIndex=\" << outputIndex << \" sum=\" << sum << \" output[outputIndex]=\" <<\n"
    "//                        output[outputIndex] << endl;\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    return output;\n"
    "}\n"
    "\n"
    "#endif\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "convolve_tilemode_float", options, "cl/forwardTiled.cl");
    // [[[end]]]
}

