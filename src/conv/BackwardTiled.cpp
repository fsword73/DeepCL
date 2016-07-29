#include "BackwardTiled.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"


#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackwardTiled::~BackwardTiled() {
    delete kernel;
//    delete broadcastMultiply;
//    delete applyActivationDeriv;
}
VIRTUAL void BackwardTiled::backward(int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
        CLWrapper *gradInputWrapper) {
    StatefulTimer::instance()->timeCheck("BackwardTiled start");

    kernel
       ->in(batchSize)
        ->in(gradOutputWrapper)
       ->in(weightsWrapper)
        ->out(gradInputWrapper);

	//ToDo
	int workgroupsize = 64;
	const int WAVE_SIZE = 64; 	//max of AMD: 64 and NV:  32 
	int workgroupPerImage =1;
	if (dim.inputSize > 19)
	{
		workgroupsize = 256;
		if (dim.filterSize == 3 && dim.inputSize > 32)
		{
			//16x64 tile
			const int HSIZE = 16;
			const int VSIZE = 64;
			workgroupPerImage = ((dim.inputSize + HSIZE - 1) / HSIZE) * ((dim.inputSize + VSIZE - 1) / VSIZE);
		}
		else
		{
			//32x32 tile
			const int HSIZE = 32;
			const int VSIZE = 32;
			workgroupPerImage = ((dim.inputSize + HSIZE - 1) / HSIZE) * ((dim.inputSize + VSIZE - 1) / VSIZE);
		}
	}
	else
	{
		int workgroupsize = (dim.inputSizeSquared / 2 + WAVE_SIZE - 1) & (!(WAVE_SIZE - 1));
		if (workgroupsize < WAVE_SIZE)
			workgroupsize = WAVE_SIZE;
		workgroupPerImage = 1;
	}


	int globalSize = batchSize * dim.numInputPlanes * workgroupPerImage * workgroupsize;
    kernel->run_1d(globalSize, workgroupsize);

    cl->finish();
    StatefulTimer::instance()->timeCheck("BackwardTiled after first kernel");

//    applyActivationDeriv->in(batchSize * dim.inputCubeSize)->in(gradInputWrapper)->in(inputDataWrapper);
//    applyActivationDeriv->run_1d(globalSize, workgroupsize);
//    cl->finish();
//    StatefulTimer::instance()->timeCheck("BackwardTiled after applyActivationDeriv");
    
    StatefulTimer::instance()->timeCheck("BackwardTiled end");
}
BackwardTiled::BackwardTiled(EasyCL *cl, LayerDimensions dim) :
        Backward(cl, dim)
            {
    std::string options = dim.buildOptionsString();
    options += ""; // " -D " + upstreamFn->getDefineName();

	//ToDo
	if (dim.inputSize > 19)
	{
		if (dim.filterSize == 3 && dim.inputSize > 32)
		{   //16x64 tile
			options += " -D TILE_WIDTH=16";
			options += " -D TILE_HEIGHT=16";
			options += " -D VTILE_REPEAT=4";
			options += " -D FIXED_WORKGROUP_SIZE=256";
		}
		else
		{   //32x32 tile: LDS bank = 32
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
		int FIXED_WORKGROUP_SIZE = (dim.inputSizeSquared / 2 + WAVE_SIZE - 1) & (!(WAVE_SIZE - 1));
		if(FIXED_WORKGROUP_SIZE < 64)
			FIXED_WORKGROUP_SIZE = 64;
		int VTILE_HEGIHT = FIXED_WORKGROUP_SIZE / dim.inputSize;
		int VTILE_REPEAT = (dim.inputSizeSquared + FIXED_WORKGROUP_SIZE - 1) / FIXED_WORKGROUP_SIZE;

		options += " -D TILE_WIDTH=" + toString(dim.inputSize);
		options += " -D TILE_HEIGHT=" + toString(VTILE_HEGIHT);
		options += " -D VTILE_REPEAT=" + toString(VTILE_REPEAT);
		options += " -D FIXED_WORKGROUP_SIZE=" + toString(FIXED_WORKGROUP_SIZE);
	}

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/backwardTiled.cl", "calcGradInput_TileMode", 'options')
    // ]]]
    // generated using cog, from cl/backwardTiled.cl:
    const char * kernelSource =  
    "\n"
    "\n"
    "// expected defines:\n"
    "//  - none\n"
    "\n"
    "// globalid as: [n][upstreamPlane][upstreamrow][upstreamcol]\n"
    "// inputdata: [n][upstreamPlane][upstreamrow][upstreamcol] 128 * 32 * 19 * 19 * 4 = 6MB\n"
    "// gradOutput: [n][outPlane][outRow][outCol] 128 * 32 * 19 * 19 * 4 = 6MB\n"
    "// weights: [filterId][inputPlane][filterRow][filterCol] 32 * 32 * 5 * 5 * 4 = 409KB\n"
    "\n"
    "#if 0\n"
    "#define  gInputSize 		64\n"
    "#define  gInputSizeSquared (gInputSize*gInputSize)\n"
    "#define  gInputPlanes  	8\n"
    "#define  gMargin 				0\n"
    "#define  gOutputSize 		64\n"
    "#define  gNumFilters    8\n"
    "#define  gFilterSize    3\n"
    "#define  gHalfFilterSize ( gFilterSize >>1)\n"
    "#define  FIXED_WORKGROUP_SIZE 64\n"
    "#endif\n"
    "#if 0\n"
    "//Following 4 defines will be depends on imageSize\n"
    "//For example: 19x19 will be\n"
    "//      TILE_WIDTH   						19\n"
    "//      TILE_HEIGHT   					10\n"
    "//      VTILE_REPEAT  					2\n"
    "//      FIXED_WORKGROUP_SIZE		192\n"
    "//\n"
    "\n"
    "//  For example: 14x14 will be\n"
    "//      TILE_WIDTH   						14\n"
    "//      TILE_HEIGHT   					4 				# 4 rows per Workgroup\n"
    "//      VTILE_REPEAT  					4\n"
    "//      FIXED_WORKGROUP_SIZE		64\n"
    "\n"
    "\n"
    "#define TILE_WIDTH   16\n"
    "#define TILE_HEIGHT  16\n"
    "\n"
    "//Calculate 2 Pixels per thread o reuse  S_Load_dwordx4 twice\n"
    "#define VTILE_REPEAT 4\n"
    "\n"
    "#define FIXED_WORKGROUP_SIZE  256\n"
    "#endif\n"
    "\n"
    "#define HTILES ((gInputSize + TILE_WIDTH-1) / TILE_WIDTH )\n"
    "#define VTILES ((gInputSize + TILE_HEIGHT-1) / (TILE_HEIGHT * VTILE_REPEAT))\n"
    "\n"
    "//Shared Memory with Odd Stride to avoid Bank Conflicts\n"
    "#define HTILE_LOCAL_SIZE ( (gFilterSize-1 + TILE_WIDTH) )\n"
    "#define VTILE_LOCAL_SIZE ( gFilterSize -1 + TILE_HEIGHT * VTILE_REPEAT)\n"
    "\n"
    "\n"
    "#define ROWS_PER_WORKGROUP    ( FIXED_WORKGROUP_SIZE / HTILE_LOCAL_SIZE )\n"
    "//For example, 19x19 split into 19x10x2, 192 threads, 2 threads are invalid\n"
    "#define MAX_VALID_ID          ( (FIXED_WORKGROUP_SIZE /TILE_WIDTH ) * TILE_WIDTH)\n"
    "\n"
    "// each time  it fetches   rows == ROWS_PER_WORKGROUP .\n"
    "#define IMAGE_LOAD_ITERATIONS ( (HTILE_LOCAL_SIZE + ROWS_PER_WORKGROUP-1) / ROWS_PER_WORKGROUP )\n"
    "\n"
    "\n"
    "#if 0\n"
    "void kernel test_kernel(__global const float* filter,\n"
    "						__global const float* gradOutput,\n"
    "						__global const float* weights,\n"
    "						__global const float* dataBuf3,\n"
    "						__global const float* dataBuf4,\n"
    "						__global const float* dataBuf5,\n"
    "						__global float* gradInput,\n"
    "						const int batchSize,\n"
    "						const int const2,\n"
    "						const int const3,\n"
    "						const int const4,\n"
    "						const int const5,\n"
    "						const int const6)\n"
    "#else\n"
    "void kernel calcGradInput_TileMode(\n"
    "const int batchSize,\n"
    "global const float *gradOutput, global const float *weights, global float *gradInput)\n"
    "#endif\n"
    "{\n"
    "		int localId = get_local_id(0);\n"
    "		int groupId = get_group_id(0);\n"
    "\n"
    "		const int upstreamImage2dId = groupId / (HTILES * VTILES) ;\n"
    "	  int n  						= upstreamImage2dId / gInputPlanes;\n"
    "    int upstreamPlane = upstreamImage2dId % gInputPlanes;\n"
    "\n"
    "\n"
    "\n"
    "		float sum = 0;\n"
    "	  float sum2 = 0;\n"
    "		float sum3 = 0;\n"
    "	  float sum4 = 0;\n"
    "		__local float __gradOutput[HTILE_LOCAL_SIZE * VTILE_LOCAL_SIZE];\n"
    "\n"
    "\n"
    "\n"
    "	  int weightPlaneOffset = upstreamPlane * gFilterSize*gFilterSize;\n"
    "		int outputImageOffset = n * gNumFilters *  (gOutputSize *  gOutputSize);\n"
    "\n"
    "    //Step1: calculuating output\n"
    "\n"
    "\n"
    "		const int tile_BL_Y = ((groupId % (HTILES * VTILES)) / HTILES) * (VTILE_REPEAT * TILE_HEIGHT);\n"
    "		const int tile_BL_X = ((groupId % (HTILES * VTILES)) % HTILES) * TILE_WIDTH;\n"
    "\n"
    "		int localRow  = localId / TILE_WIDTH ;\n"
    "		int localCol  = localId % TILE_WIDTH ;\n"
    "\n"
    "		int upstreamRow = localRow  + tile_BL_Y ;\n"
    "		int upstreamCol	= localCol  + tile_BL_X ;\n"
    "\n"
    "		bool bValid =  upstreamRow < gInputSize && upstreamCol < gInputSize && localId < MAX_VALID_ID;\n"
    "\n"
    "		//2nd pixel vaid or not\n"
    "		bool bValid2 =  (upstreamRow + TILE_HEIGHT*1) < gInputSize && upstreamCol < gInputSize && localId < MAX_VALID_ID;\n"
    "		bool bValid3 =  (upstreamRow + TILE_HEIGHT*2) < gInputSize && upstreamCol < gInputSize && localId < MAX_VALID_ID;\n"
    "		bool bValid4 =  (upstreamRow + TILE_HEIGHT*3) < gInputSize && upstreamCol < gInputSize && localId < MAX_VALID_ID;\n"
    "\n"
    "		upstreamRow += gMargin;\n"
    "		upstreamCol	+= gMargin;\n"
    "\n"
    "		//  Filter   [  0,0   0,1   0,2]  Pixel  [   0,0    0,-1    0,-2]\n"
    "		//           [  1,0   1,1   1,2]         [  -1,0   -1,-1   -1,-2]\n"
    "		//           [  2,0   2,1   2,2]         [  -2,0   -2,-1   -2,-2]\n"
    "\n"
    "\n"
    "		//Step2: loading_offset;\n"
    "    int local_x, local_y;\n"
    "		bool bImageLoad = false;   //possible to Load\n"
    "		bool bImageValidX = false;  //Really have to Load\n"
    "		int image_x, image_y;\n"
    "		if( localId < (ROWS_PER_WORKGROUP * HTILE_LOCAL_SIZE))\n"
    "		{\n"
    "			  bImageLoad = true;\n"
    "				local_x	= localId % HTILE_LOCAL_SIZE;\n"
    "			  local_y = localId / HTILE_LOCAL_SIZE;\n"
    "\n"
    "			  //shift left to (gFilterSize-1)\n"
    "			  image_x = local_x + tile_BL_X - (gFilterSize-1) ;\n"
    "			  image_y = local_y + tile_BL_Y - (gFilterSize-1) ;\n"
    "				if( image_x >= 0 && image_x < gOutputSize)\n"
    "				{\n"
    "						bImageValidX = true;\n"
    "				}\n"
    "				image_x +=  gMargin;\n"
    "				image_y +=  gMargin;\n"
    "		}\n"
    "\n"
    "\n"
    "		// aggregate over [outPlane][outRow][outCol]\n"
    "		int forceloop = min(gNumFilters, gNumFilters*batchSize);\n"
    "\n"
    "    for (int outPlane = 0; outPlane < forceloop; outPlane++) {\n"
    "\n"
    "\n"
    "			   if(bImageLoad)\n"
    "				 {\n"
    "						for(int i = 0; i < IMAGE_LOAD_ITERATIONS; i++)\n"
    "						{\n"
    "							int localOffset =	(local_y + i * ROWS_PER_WORKGROUP)  * HTILE_LOCAL_SIZE + local_x;\n"
    "\n"
    "							//Boundary check\n"
    "							if( localOffset < (HTILE_LOCAL_SIZE * VTILE_LOCAL_SIZE))\n"
    "							{\n"
    "									float value;\n"
    "									int row = 0;\n"
    "									bool bProcess = false;\n"
    "\n"
    "									row = image_y + i * ROWS_PER_WORKGROUP;\n"
    "\n"
    "									bProcess = row >= 0 && row < gOutputSize && bImageValidX ;\n"
    "\n"
    "									int offset =  outputImageOffset +\n"
    "																	outPlane * gOutputSize * gOutputSize +\n"
    "																	row * gOutputSize +\n"
    "																	image_x;\n"
    "\n"
    "									 value =  gradOutput[offset];\n"
    "									 if(!bProcess){\n"
    "											value = 0;\n"
    "										}\n"
    "\n"
    "									__gradOutput[localOffset] = value;\n"
    "							}\n"
    "						}\n"
    "				 }\n"
    "\n"
    "\n"
    "				barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#if (gFilterSize > 5)\n"
    "				if(bValid){\n"
    "					//No negative offset mode, it is slow code\n"
    "\n"
    "					//LLVM with Pointers to produce less address\n"
    "					//int thisWeightIndex = (( outPlane * gInputPlanes\n"
    "					// 											+ upstreamPlane) * gFilterSize\n"
    "					//											+ filterRow) * gFilterSize\n"
    "					//            					+ filterCol;\n"
    "\n"
    "					__global const float *f = &weights[weightPlaneOffset + outPlane * gInputPlanes * gFilterSize *gFilterSize];\n"
    "					for(int i=0; i < gFilterSize; i++)\n"
    "					{\n"
    "						for(int j=0; j < gFilterSize; j++)\n"
    "						{\n"
    "								//shift local memory (gFilterSize-1) to image_xy\n"
    "								int outRow = localRow - i + (gFilterSize-1);\n"
    "								int outCol = localCol - j + (gFilterSize-1);\n"
    "\n"
    "								int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;\n"
    "\n"
    "								float thisError = __gradOutput[resultIndex];\n"
    "								float thisWeight = *(f + i * gFilterSize + j);\n"
    "								sum += thisError* thisWeight;\n"
    "						}\n"
    "\n"
    "\n"
    "						//Reduce constant loading\n"
    "						if(bValid2)\n"
    "						for(int j=0; j < gFilterSize; j++)\n"
    "						{\n"
    "									//shift local memory (gFilterSize-1) to image_xy\n"
    "								int outRow = localRow - i + (gFilterSize-1) + TILE_HEIGHT ;\n"
    "								int outCol = localCol - j + (gFilterSize-1);\n"
    "\n"
    "									int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;\n"
    "\n"
    "									float thisError = __gradOutput[resultIndex];\n"
    "									float thisWeight = *(f + i * gFilterSize + j);\n"
    "									sum2 += thisError* thisWeight;\n"
    "						}\n"
    "						if(bValid3)\n"
    "						for(int j=0; j < gFilterSize; j++)\n"
    "						{\n"
    "									//shift local memory (gFilterSize-1) to image_xy\n"
    "									int outRow = localRow - i + (gFilterSize-1) + TILE_HEIGHT *2 ;\n"
    "									int outCol = localCol - j + (gFilterSize-1);\n"
    "\n"
    "									int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;\n"
    "\n"
    "									float thisError = __gradOutput[resultIndex];\n"
    "									float thisWeight = *(f + i * gFilterSize + j);\n"
    "									sum3 += thisError* thisWeight;\n"
    "						}\n"
    "						if(bValid4)\n"
    "						for(int j=0; j < gFilterSize; j++)\n"
    "						{\n"
    "									//shift local memory (gFilterSize-1) to image_xy\n"
    "									int outRow = localRow - i + (gFilterSize-1) + TILE_HEIGHT *3 ;\n"
    "									int outCol = localCol - j + (gFilterSize-1);\n"
    "\n"
    "									int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;\n"
    "\n"
    "									float thisError = __gradOutput[resultIndex];\n"
    "									float thisWeight = *(f + i * gFilterSize + j);\n"
    "									sum4 += thisError* thisWeight;\n"
    "						}\n"
    "					}\n"
    "				}\n"
    "#else\n"
    "				if(bValid){\n"
    "					//No negative offset mode, it is slow code\n"
    "						for(int i=0; i < gFilterSize; i++)\n"
    "						{\n"
    "							for(int j=0; j < gFilterSize; j++)\n"
    "							{\n"
    "									//shift local memory (gFilterSize-1) to image_xy\n"
    "									int outRow = localRow - i + (gFilterSize-1);\n"
    "									int outCol = localCol - j + (gFilterSize-1);\n"
    "\n"
    "									int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;\n"
    "\n"
    "									float thisError = __gradOutput[resultIndex];\n"
    "									int thisWeightIndex = outPlane * gInputPlanes* gFilterSize *gFilterSize\n"
    "																				+ weightPlaneOffset +\n"
    "																				+ i * gFilterSize\n"
    "																				+ j;\n"
    "									float thisWeight = weights[thisWeightIndex];\n"
    "									sum += thisError* thisWeight;\n"
    "							}\n"
    "						}\n"
    "						if(bValid2){\n"
    "						for(int i=0; i < gFilterSize; i++)\n"
    "							for(int j=0; j < gFilterSize; j++)\n"
    "								{\n"
    "\n"
    "										//shift local memory (gFilterSize-1) to image_xy\n"
    "										int outRow = localRow - i + (gFilterSize-1) + TILE_HEIGHT;\n"
    "										int outCol = localCol - j + (gFilterSize-1);\n"
    "\n"
    "										int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;\n"
    "\n"
    "										float thisError = __gradOutput[resultIndex];\n"
    "									int thisWeightIndex = outPlane * gInputPlanes* gFilterSize *gFilterSize\n"
    "																				+ weightPlaneOffset +\n"
    "																				+ i * gFilterSize\n"
    "																				+ j;\n"
    "										float thisWeight = weights[thisWeightIndex];\n"
    "										sum2 += thisError* thisWeight;\n"
    "								}\n"
    "						}\n"
    "\n"
    "						if(bValid3){\n"
    "						for(int i=0; i < gFilterSize; i++)\n"
    "							for(int j=0; j < gFilterSize; j++)\n"
    "									{\n"
    "										//shift local memory (gFilterSize-1) to image_xy\n"
    "										int outRow = localRow - i + (gFilterSize-1) + TILE_HEIGHT*2;\n"
    "										int outCol = localCol - j + (gFilterSize-1);\n"
    "\n"
    "										int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;\n"
    "\n"
    "										float thisError = __gradOutput[resultIndex];\n"
    "									int thisWeightIndex = outPlane * gInputPlanes* gFilterSize *gFilterSize\n"
    "																				+ weightPlaneOffset +\n"
    "																				+ i * gFilterSize\n"
    "																				+ j;\n"
    "										float thisWeight = weights[thisWeightIndex];\n"
    "										sum3 += thisError* thisWeight;\n"
    "								}\n"
    "						}\n"
    "\n"
    "						if(bValid4){\n"
    "						for(short i=0; i < gFilterSize; i++)\n"
    "							for(short j=0; j < gFilterSize; j++)\n"
    "								{\n"
    "										//shift local memory (gFilterSize-1) to image_xy\n"
    "										int outRow = localRow - i + (gFilterSize-1) + TILE_HEIGHT*3;\n"
    "										int outCol = localCol - j + (gFilterSize-1);\n"
    "\n"
    "										int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;\n"
    "\n"
    "										float thisError = __gradOutput[resultIndex];\n"
    "									int thisWeightIndex = outPlane * gInputPlanes* gFilterSize *gFilterSize\n"
    "																				+ weightPlaneOffset +\n"
    "																				+ i * gFilterSize\n"
    "																				+ j;\n"
    "										float thisWeight = weights[thisWeightIndex];\n"
    "										sum4 += thisError* thisWeight;\n"
    "								}\n"
    "						}\n"
    "				}\n"
    "#endif\n"
    "\n"
    "\n"
    "    }//while\n"
    "\n"
    "		if (bValid)\n"
    "		{\n"
    "\n"
    "		   unsigned	int Offset  = upstreamImage2dId * gInputSizeSquared + upstreamRow * gInputSize + upstreamCol;\n"
    "\n"
    "        gradInput[Offset] = sum;\n"
    "			 if(bValid2)\n"
    "			 {\n"
    "				 gradInput[Offset + TILE_HEIGHT * gInputSize] = sum2;\n"
    "			 }\n"
    "			 if(bValid3)\n"
    "			 {\n"
    "				 gradInput[Offset + TILE_HEIGHT * 2 * gInputSize] = sum3;\n"
    "			 }\n"
    "			 if(bValid4)\n"
    "			 {\n"
    "				 gradInput[Offset + TILE_HEIGHT * 3 * gInputSize] = sum4;\n"
    "			 }\n"
    "\n"
    "    }\n"
    "\n"
    "}\n"
    "\n"
    "\n"
    "\n"
    "#if 0\n"
    "void kernel calcGradInput(\n"
    "        const int batchSize,\n"
    "        global const float *gradOutput, global float *weights, global float *gradInput) {\n"
    "    int globalId = get_global_id(0);\n"
    "\n"
    "    const int upstreamImage2dId = globalId / gInputSizeSquared;\n"
    "\n"
    "    const int intraImageOffset = globalId % gInputSizeSquared;\n"
    "    const int upstreamRow = intraImageOffset / gInputSize;\n"
    "    const int upstreamCol = intraImageOffset % gInputSize;\n"
    "\n"
    "    const int upstreamPlane = upstreamImage2dId % gInputPlanes;\n"
    "    const int n = upstreamImage2dId / gInputPlanes;\n"
    "\n"
    "    if (n >= batchSize) {\n"
    "        return;\n"
    "    }\n"
    "\n"
    "    const int minFilterRow = max(0, upstreamRow + gMargin - (gOutputSize - 1));\n"
    "    const int maxFilterRow = min(gFilterSize - 1, upstreamRow + gMargin);\n"
    "    const int minFilterCol = max(0, upstreamCol + gMargin - (gOutputSize -1));\n"
    "    const int maxFilterCol = min(gFilterSize - 1, upstreamCol + gMargin);\n"
    "\n"
    "    float sumWeightTimesOutError = 0;\n"
    "    // aggregate over [outPlane][outRow][outCol]\n"
    "    for (int outPlane = 0; outPlane < gNumFilters; outPlane++) {\n"
    "        for (int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++) {\n"
    "            int outRow = upstreamRow + gMargin - filterRow;\n"
    "            for (int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++) {\n"
    "                int outCol = upstreamCol + gMargin - filterCol;\n"
    "                int resultIndex = (( n * gNumFilters\n"
    "                          + outPlane) * gOutputSize\n"
    "                          + outRow) * gOutputSize\n"
    "                          + outCol;\n"
    "                float thisError = gradOutput[resultIndex];\n"
    "                int thisWeightIndex = (( outPlane * gInputPlanes\n"
    "                                    + upstreamPlane) * gFilterSize\n"
    "                                    + filterRow) * gFilterSize\n"
    "                                    + filterCol;\n"
    "                float thisWeight = weights[thisWeightIndex];\n"
    "                float thisWeightTimesError = thisWeight * thisError;\n"
    "                sumWeightTimesOutError += thisWeightTimesError;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    gradInput[globalId] = sumWeightTimesOutError;\n"
    "}\n"
    "\n"
    "CPU source code\n"
    "    for(int n = 0; n < batchSize; n++) {\n"
    "        for(int upstreamPlane = 0; upstreamPlane < dim.inputPlanes; upstreamPlane++) {\n"
    "            for(int upstreamRow = 0; upstreamRow < dim.inputSize; upstreamRow++) {\n"
    "                int minFilterRow = std::max(0, upstreamRow + margin - (dim.outputSize - 1));\n"
    "                int maxFilterRow = std::min(dim.filterSize - 1, upstreamRow + margin);\n"
    "                for(int upstreamCol = 0; upstreamCol < dim.inputSize; upstreamCol++) {\n"
    "                    float sumWeightTimesGradOutput = 0;\n"
    "                    // aggregate over [outPlane][outRow][outCol]\n"
    "                    int minFilterCol = std::max(0, upstreamCol + margin - (dim.outputSize -1));\n"
    "                    int maxFilterCol = std::min(dim.filterSize - 1, upstreamCol + margin);\n"
    "                    for(int outPlane = 0; outPlane < dim.numFilters; outPlane++) {\n"
    "                        for(int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++) {\n"
    "                            int outRow = upstreamRow + margin - filterRow;\n"
    "                            for(int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++) {\n"
    "                                int outCol = upstreamCol + margin - filterCol;\n"
    "                                int resultIndex = (( n\n"
    "                                    * dim.numFilters + outPlane)\n"
    "                                    * dim.outputSize + outRow)\n"
    "                                    * dim.outputSize + outCol;\n"
    "                                float thisGradOutput = gradOutput[resultIndex];\n"
    "                                int thisWeightIndex = (( outPlane\n"
    "                                    * dim.inputPlanes + upstreamPlane)\n"
    "                                    * dim.filterSize + filterRow)\n"
    "                                    * dim.filterSize + filterCol;\n"
    "                                float thisWeight = weights[thisWeightIndex];\n"
    "                                sumWeightTimesGradOutput += thisWeight * thisGradOutput;\n"
    "                            }\n"
    "                        }\n"
    "                    }\n"
    "                    int inputIndex = (( n\n"
    "                        * dim.inputPlanes + upstreamPlane)\n"
    "                        * dim.inputSize + upstreamRow)\n"
    "                        * dim.inputSize + upstreamCol;\n"
    "                    gradInput[inputIndex] = sumWeightTimesGradOutput; // * activationDerivativeUpstream;\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "\n"
    "\n"
    "#endif\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "calcGradInput_TileMode", options, "cl/backwardTiled.cl");
    // [[[end]]]

}

