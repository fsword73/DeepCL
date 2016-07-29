// images are organized like [imageId][plane][row][col]
// filters are organized like [filterid][inplane][filterrow][filtercol]
// output are organized like [imageid][filterid][row][col]
// global group id is organized like output, ie: [imageid][outplane][HTILE_ID][VTILE_ID]


#if 0 //REAL ARGS 
#define BIASED 
#define gNumInputPlanes 1 
#define gInputPlanes 1 
#define gInputSize 28 
#define gInputSizeSquared 784 
#define gNumFilters 8 
#define gFilterSize 5 
#define gHalfFilterSize 2 
#define gFilterSizeSquared 25 
#define gNumOutputPlanes 8 
#define gOutputPlanes 150 
#define gOutputSize 28 
#define gOutputSizeSquared (28*28)
#define gPadZeros 1 
#define gMargin 2 
#define gEven 0 
#define gSkip 0 
#define TILE_WIDTH 32 
#define TILE_HEIGHT 8 
#define VTILE_REPEAT 4 
#define FIXED_WORKGROUP_SIZE 256
#endif
#if 0
#define gOutputSize 64
#define gOutputSizeSquared ( gOutputSize * gOutputSize)
#define gNumFilters 8
#define gNumInputPlanes 8
#define gInputSize 64 
#define gInputSizeSquared (gInputSize *gInputSize)
#define gFilterSize  3
#define gHalfFilterSize  ( gFilterSize >>1)
#define gFilterSizeSquared (gFilterSize *gFilterSize )
#define gEven 0

//Following 4 lines to be adjusted with differnt gOutputSize & gFilterSize
#define TILE_WIDTH   16
#define TILE_HEIGHT  16

//Calculate 4 Pixels per thread o reuse S_Load_dwordx4 
#define VTILE_REPEAT 4  
#define FIXED_WORKGROUP_SIZE  256
#endif
//The bottleneck of this kernel is : VALU, Scarlar, LDS,  not buffer loading 

//go: 19x19 : TILE_WIDTH = 19, TILE_HEIGHT=2 ,  TILE_REPEAT: 2 
//default: FilterSize; 3x3  : TILE_WIDTH = 16, TILE_HEIGHT = 16  TILE_REPEAT: 2 
//Other  : FilterSize; 5x5  : TILE_WIDTH = 32, TILE_HEIGHT = 8   TILE_REPEAT: 4 
//32x32     :        				: TILE_WIDTH = 32, TILE_HEIGHT = 8   TILE_REPEAT: 4         	


#define HTILES ((gOutputSize + TILE_WIDTH-1) / TILE_WIDTH )
#define VTILES ((gOutputSize + TILE_HEIGHT-1) / (TILE_HEIGHT * VTILE_REPEAT)) 
#define HTILE_LOCAL_SIZE ( gFilterSize -1 + TILE_WIDTH)
#define VTILE_LOCAL_SIZE ( gFilterSize -1 + TILE_HEIGHT * VTILE_REPEAT)


#define ROWS_PER_WORKGROUP    ( FIXED_WORKGROUP_SIZE / HTILE_LOCAL_SIZE )
#define MAX_VALID_ID          ( (FIXED_WORKGROUP_SIZE /TILE_WIDTH ) * TILE_WIDTH)

// each time  it fetches   rows == ROWS_PER_WORKGROUP . 
#define IMAGE_LOAD_ITERATIONS ( (HTILE_LOCAL_SIZE + ROWS_PER_WORKGROUP-1) / ROWS_PER_WORKGROUP )

//Load image  into LDS once 
//Load by Rows 
//Load Filters by SGPR  once

#if gOutputSize == 1	
void convolve_1x1_float(
    const int batchSize,
    global const float *inputs, global const float *filters, 
    global float *output, __local float* sdata, int globalId, int localId) 
{
    int outputImage2Id = globalId / gOutputSizeSquared;
    int exampleId = outputImage2Id / gNumFilters;
    int filterId = outputImage2Id % gNumFilters;

    // intraimage coords
    int localid = globalId % gOutputSizeSquared;
    int outputRow = localid / gOutputSize;
    int outputCol = localid % gOutputSize;

    global float const*inputCube = inputs + exampleId * gNumInputPlanes * gInputSizeSquared;
    global float const*filterCube = filters + filterId * gNumInputPlanes * gFilterSizeSquared;

    float sum = 0;
    if (exampleId < batchSize) {
#define iterations (( gNumInputPlanes + FIXED_WORKGROUP_SIZE -1 ) / FIXED_WORKGROUP_SIZE)
				for(int i= 0; i < iterations; i++)
				{
						int inputPlaneIdx = localId + i* FIXED_WORKGROUP_SIZE;
						if(inputPlaneIdx >= gNumInputPlanes)
							   break;
						
            global float const*inputPlane = inputCube + inputPlaneIdx * gInputSizeSquared;
            global float const*filterPlane = filterCube + inputPlaneIdx * gFilterSizeSquared;
            for (int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++) {
                // trying to reduce register pressure...
                #if gPadZeros == 1
                    #define inputRowIdx (outputRow + u)
                #else
                    #define inputRowIdx (outputRow + u + gHalfFilterSize)
                #endif
                global float const *inputRow = inputPlane + inputRowIdx * gInputSize;
                global float const *filterRow = filterPlane + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                bool rowOk = inputRowIdx >= 0 && inputRowIdx < gInputSize;
                #pragma unroll
                for (int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++) {
                    #if gPadZeros == 1
                        #define inputColIdx (outputCol + v)
                    #else
                        #define inputColIdx (outputCol + v + gHalfFilterSize)
                    #endif
                    bool process = rowOk && inputColIdx >= 0 && inputColIdx < gInputSize;
                    if (process) {
                            sum += inputRow[inputColIdx] * filterRow[v];
                    }
                }
            }
        }
    }
		
		//Reduction
		
		//store into local 	 
		sdata[localId] = sum;
		barrier(CLK_LOCAL_MEM_FENCE);
		for(unsigned int s = FIXED_WORKGROUP_SIZE >>1; s > 0; s >>= 1) 
		{
			if(localId < s) 
			{
				sdata[localId] += sdata[localId + s];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	
		if(localId == 0)
		{		
			output[globalId] = sdata[0];
		}
}
#endif

#if 0
void kernel test_kernel(__global const float* filter,
						__global const float* inputs,
						__global const float* filters,	
						__global const float* weights,
						__global const float* dataBuf4,
						__global const float* dataBuf5,
						__global float* output,
						const int batchSize,  
						const int const2,  
						const int const3,  
						const int const4,  
						const int const5,  
						const int const6)  
#else 
void kernel convolve_tilemode_float(
    const int batchSize,
    global const float *inputs, global const float *filters, 
    global float *output) 
#endif
{
			
		int localId = get_local_id(0);
		int groupId = get_group_id(0);
	
#if gOutputSize == 1	
		__local float sdata[FIXED_WORKGROUP_SIZE];
    convolve_1x1_float(batchSize, inputs, filters, output, sdata, groupId, localId);
#else		
		int outputImage2Id = 	groupId /(HTILES * VTILES);
    int batchId  = outputImage2Id / gNumFilters;
    int filterId = outputImage2Id % gNumFilters;
		float sum = 0;	
	  float sum2 = 0;
		float sum3 = 0;
	  float sum4 = 0;
		__local float __inputs[HTILE_LOCAL_SIZE * VTILE_LOCAL_SIZE];
			
	  int filterIdOffset = filterId * gNumInputPlanes * gFilterSize * gFilterSize;

		
		//Step1: calculuating output
		int tile_BL_X, tile_BL_Y;	
		tile_BL_Y = ((groupId % (HTILES * VTILES)) / HTILES) * (TILE_HEIGHT* VTILE_REPEAT);
		tile_BL_X = ((groupId % (HTILES * VTILES)) % HTILES) * TILE_WIDTH;

		//Offset in Local Memory 
		int localRow  = localId / TILE_WIDTH ;
		int localCol  = localId % TILE_WIDTH ;


		//Offset in current Memory
		int outputRow = localId / TILE_WIDTH  + tile_BL_Y; 
		int outputCol	= localId % TILE_WIDTH  + tile_BL_X;
		
		bool bValid =  outputRow < gOutputSize && outputCol < gOutputSize && localId < MAX_VALID_ID;
		
		//2nd pixel vaid or not		
		bool bValid2 =  (outputRow + TILE_HEIGHT)   < gOutputSize && outputCol < gOutputSize && localId < MAX_VALID_ID;
		bool bValid3 =  (outputRow + TILE_HEIGHT*2) < gOutputSize && outputCol < gOutputSize && localId < MAX_VALID_ID;
		bool bValid4 =  (outputRow + TILE_HEIGHT*3) < gOutputSize && outputCol < gOutputSize && localId < MAX_VALID_ID;
		
		//[Batch][InputPlane][Row][col]
		int inputImageOffset = batchId * gNumInputPlanes * gInputSizeSquared ;
		
		//Step2: loading_offset;
    int local_x, local_y;
		bool bImageLoad 	= false;
		bool bValidX = false; 
		int image_x, image_y;
		
		//only partial threads to load Input 
		if( localId < (ROWS_PER_WORKGROUP * HTILE_LOCAL_SIZE))
		{
			  bImageLoad = true;
				//Local memory Offset 
				local_x	= localId % HTILE_LOCAL_SIZE;
			  local_y = localId / HTILE_LOCAL_SIZE;
			  
			  //load Image Offset 
			  image_x = local_x + tile_BL_X;
			  image_y = local_y + tile_BL_Y;

				//reduce operation for inner loop
			  bValidX = image_x >=0 && image_x < gInputSize; 
			
#if gPadZeros == 1
			  image_x -= gHalfFilterSize;
			  image_y -= gHalfFilterSize;				
#endif 			

			
		}
		

		
		//step3: Fordward 		
		int forceloop = min(gNumInputPlanes, gNumInputPlanes*batchSize);
		for (int inputPlaneIdx = 0; inputPlaneIdx < forceloop; inputPlaneIdx++) {
			  
			  //only bImageLoad == true will load images  
			  if(bImageLoad)
				{							
						for(int i = 0; i < IMAGE_LOAD_ITERATIONS; i++)
						{
								int localOffset =	(local_y + i * ROWS_PER_WORKGROUP)  * HTILE_LOCAL_SIZE + local_x;
								
								//boundary check
								if( localOffset < (HTILE_LOCAL_SIZE * VTILE_LOCAL_SIZE))
								{							  
										float value;				
										int row = 0;
										bool bProcess = false;

										row = image_y + i * ROWS_PER_WORKGROUP; 						
										bProcess = row >= 0 && row < gInputSize && bValidX;
										unsigned inputOffset =  inputImageOffset + 
																		 inputPlaneIdx * gInputSizeSquared +
																		 row * gInputSize + 
																		 image_x;
									 value =  inputs[inputOffset];
										if( !bProcess)
										{
											value = 0;
										}
										
										__inputs[localOffset] = value;
								}			
						}
				}

				barrier(CLK_LOCAL_MEM_FENCE);
				
				// SUM += 				
#if (gFilterSize>5)				
				//minimized loading of constant for big gFilterSize				
				if(bValid)
				{
						  //Fix the LLVM performance bug for address calculuating
						  //int filterIdx  = 	filterIdOffset + inputPlaneIdx * gFilterSizeSquared +  i * gFilterSize + j;
							__global const *f = &filters[filterIdOffset + inputPlaneIdx * gFilterSizeSquared];
					
							for(int i = 0; i < gFilterSize; i++)
							{
									for(int j =0; j < gFilterSize; j++)
									{
											//image COL 0, 1 ,2 * filter COL 0, 1, 2 											 
											unsigned int inputIdx   = (localRow +  i) * HTILE_LOCAL_SIZE + 
																								 (localCol  + j);
										  float thisWeight = *(f + i * gFilterSize + j); 
											sum += __inputs[inputIdx] * thisWeight;
									}					
									if(bValid2)
									for(int j =0; j < gFilterSize; j++)
									{
											
											unsigned int inputIdx   = (localRow +  i + TILE_HEIGHT) * HTILE_LOCAL_SIZE + 
																								(localCol + j);
										
											float thisWeight = *(f + i * gFilterSize + j); 
											sum2 += __inputs[inputIdx] * thisWeight;
									}	
									if(bValid3)
									for(int j =0; j < gFilterSize; j++)
									{
											
											unsigned int inputIdx   = (localRow  + i + TILE_HEIGHT*2) * HTILE_LOCAL_SIZE + 
																			 (localCol  + j);

										float thisWeight = *(f + i * gFilterSize + j); 
										sum3 += __inputs[inputIdx] * thisWeight;
									}	

									if(bValid4)
									for(int j =0; j < gFilterSize; j++)
									{
											unsigned int inputIdx   = (localRow  + i + TILE_HEIGHT*3) * HTILE_LOCAL_SIZE + 
																			 (localCol  + j);
											float thisWeight = *(f + i * gFilterSize + j); 
											sum4 += __inputs[inputIdx] * thisWeight;
									}	
						}
			 }
#else
				//Minimized Scalar Instructions
				if(bValid)
				{
							for( int i = 0; i < gFilterSize; i++)
							{
								for(int j =0; j < gFilterSize; j++)
								{
										//image COL 0, 1 ,2 * filter COL 0, 1, 2
										int inputIdx   = (localRow  + i) * HTILE_LOCAL_SIZE + 
																		 (localCol  + j);
									
									  int filterIdx  = filterIdOffset + 
																				inputPlaneIdx * gFilterSizeSquared + 
																			i * gFilterSize + 
																			j;

										
										
										sum += __inputs[inputIdx] * filters[filterIdx];
								}					
							}
								if(bValid2){
									for( int i = 0; i < gFilterSize; i++)
										for(int j =0; j < gFilterSize; j++)
										{
												int inputIdx   =  (localRow + i +TILE_HEIGHT) * HTILE_LOCAL_SIZE + 
																					(localCol + j);
											
												int filterIdx  = 	filterIdOffset + 
																					inputPlaneIdx * gFilterSizeSquared + 
																					i * gFilterSize + 
																					j;
												
												sum2 += __inputs[inputIdx] * filters[filterIdx];
										}					
								}							
								if(bValid3){
									for( int i = 0; i < gFilterSize; i++)
										for(int j =0; j < gFilterSize; j++)
										{
												//shift gHalfFilterSize
												int inputIdx   =  (localRow + i +TILE_HEIGHT*2) * HTILE_LOCAL_SIZE + 
																					(localCol + j);
											
												int filterIdx  = 	filterIdOffset + 
																					inputPlaneIdx * gFilterSizeSquared + 
																					i * gFilterSize + 
																					j;
												
												sum3 += __inputs[inputIdx] * filters[filterIdx];
										}					
								}							
								if(bValid4){
									for( int i = 0; i < gFilterSize; i++)
										for(int j =0; j < gFilterSize; j++)
										{
												//shift gHalfFilterSize
												int inputIdx   =  (localRow + i +TILE_HEIGHT*3) * HTILE_LOCAL_SIZE + 
																					(localCol + j);
											
												int filterIdx  = 	filterIdOffset + 
																					inputPlaneIdx * gFilterSizeSquared + 
																					i * gFilterSize + 
																					j;
												
												sum4 += __inputs[inputIdx] * filters[filterIdx];
										}					
								}							

				}				
#endif
		}	
			

		if (bValid) 
		{

		   unsigned int outputOffset  = outputImage2Id * gOutputSizeSquared + outputRow * gOutputSize + outputCol;
			
       output[outputOffset] = sum;
			 if(bValid2)
			 {
				 output[outputOffset + TILE_HEIGHT * gOutputSize] = sum2;
			 }
			 if(bValid3)
			 {
				 output[outputOffset + TILE_HEIGHT * 2 * gOutputSize] = sum3;
			 }
			 if(bValid4)
			 {
				 output[outputOffset + TILE_HEIGHT * 3 * gOutputSize] = sum4;
			 }
			 
    }
#endif		
}




#if 0
VIRTUAL float *ForwardCpu::forward(int batchSize, float *inputData, float *weights, float *bias) {
//    cout << "ForwardCpu::forward outputcubesize=" << dim.outputCubeSize << " batchSize=" << batchSize << endl;
    float *output = new float[ dim.outputCubeSize * batchSize ];
    for(int n = 0; n < batchSize; n++) {
        for(int filter = 0; filter < dim.numFilters; filter++) {
            for(int outRow = 0; outRow < dim.outputSize; outRow += 1 + dim.skip) {
                for(int outCol = 0; outCol < dim.outputSize; outCol += 1 + dim.skip) {
                    float sum = 0;
                    for(int inPlane = 0; inPlane < dim.inputPlanes; inPlane++) {
//                        cout << "inplane=" << inPlane << endl;
                        for(int u = -dim.halfFilterSize; u <= dim.halfFilterSize; u++) {
                            int inRow = outRow * (dim.skip + 1) + u + (dim.padZeros ? 0 : dim.halfFilterSize);
//                                cout << "candidate inRow " << inRow << endl;
                            if(inRow < 0 || inRow > dim.inputSize - 1) {
                                continue;
                            }
                            int filterRow = u + dim.halfFilterSize;
                            for(int v = -dim.halfFilterSize; v <= dim.halfFilterSize; v++) {
                                int inCol = outCol * (dim.skip + 1) + v + (dim.padZeros ? 0 : dim.halfFilterSize);
                                int filterCol = v + dim.halfFilterSize;
                                if(inCol < 0 || inCol > dim.inputSize - 1) {
                                    continue;
                                }
                                int inputIndex = (( n
                                    * dim.inputPlanes + inPlane)
                                    * dim.inputSize + inRow)
                                    * dim.inputSize + inCol;
                                int weightIndex = (( filter 
                                    * dim.inputPlanes + inPlane) 
                                    * dim.filterSize  + filterRow)
                                    * dim.filterSize  + filterCol;
//                                    cout << "inpos " << inRow << "," << inCol << " outpos " << outRow << "," << outCol
//                                        << " filterpos " << filterRow << "," << filterCol << endl;
                                float sumchange = inputData[ inputIndex] * weights[ weightIndex ];
                                if(sumchange != 0) {
//                                        cout << inputData[inputIndex] << " * " << weights[weightIndex] << " = " << sumchange << endl;
                                }
                                sum += sumchange;
//                                cout << "inputIndex=" << inputIndex << " weightIndex=" << weightIndex << 
//                                    "  inputData[inputIndex]=" << inputData[inputIndex] << " weights[weightIndex]=" << weights[weightIndex] << " sumchange " << sumchange << " sum=" << sum << endl;
                            }
                        }
                    }
                    if(dim.biased) {
                        sum += bias[filter];
                    }
//                    sum = fn->calc(sum);
                    int outputIndex = (( n 
                        * dim.numFilters + filter) 
                        * dim.outputSize + outRow)
                        * dim.outputSize + outCol;
                    output[outputIndex] = sum;
//                    cout << "outputIndex=" << outputIndex << " sum=" << sum << " output[outputIndex]=" <<
//                        output[outputIndex] << endl;
                }
            }
        }
    }
    return output;		
}

#endif