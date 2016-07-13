
// expected defines:
//  - none

// globalid as: [n][upstreamPlane][upstreamrow][upstreamcol]
// inputdata: [n][upstreamPlane][upstreamrow][upstreamcol] 128 * 32 * 19 * 19 * 4 = 6MB
// gradOutput: [n][outPlane][outRow][outCol] 128 * 32 * 19 * 19 * 4 = 6MB
// weights: [filterId][inputPlane][filterRow][filterCol] 32 * 32 * 5 * 5 * 4 = 409KB

#if 0
#define  gInputSize 		64
#define  gInputSizeSquared (gInputSize*gInputSize)
#define  gInputPlanes  	8
#define  gMargin 				0
#define  gOutputSize 		64
#define  gNumFilters    8 
#define  gFilterSize    3
#define  gHalfFilterSize ( gFilterSize >>1)


//Following 4 defines will be depends on imageSize
//For example: 19x19 will be 
//      TILE_WIDTH   						19      
//      TILE_HEIGHT   					10 
//      VTILE_REPEAT  					2
//      FIXED_WORKGROUP_SIZE		192 
//

//  For example: 14x14 will be  
//      TILE_WIDTH   						14         
//      TILE_HEIGHT   					4 				# 4 rows per Workgroup
//      VTILE_REPEAT  					4
//      FIXED_WORKGROUP_SIZE		64 


#define TILE_WIDTH   16
#define TILE_HEIGHT  16

//Calculate 2 Pixels per thread o reuse  S_Load_dwordx4 twice
#define VTILE_REPEAT 4  

#define FIXED_WORKGROUP_SIZE  256
#endif

#define HTILES ((gInputSize + TILE_WIDTH-1) / TILE_WIDTH )
#define VTILES ((gInputSize + TILE_HEIGHT-1) / (TILE_HEIGHT * VTILE_REPEAT)) 

//Shared Memory with Odd Stride to avoid Bank Conflicts
#define HTILE_LOCAL_SIZE ( ( gFilterSize + TILE_WIDTH) | 0x1)
#define VTILE_LOCAL_SIZE ( gFilterSize -1 + TILE_HEIGHT * VTILE_REPEAT)


#define ROWS_PER_WORKGROUP    ( FIXED_WORKGROUP_SIZE / HTILE_LOCAL_SIZE )
//For example, 19x19 split into 19x10x2, 192 threads, 2 threads are invalid
#define MAX_VALID_ID          ( (FIXED_WORKGROUP_SIZE /TILE_WIDTH ) * TILE_WIDTH)

// each time  it fetches   rows == ROWS_PER_WORKGROUP . 
#define IMAGE_LOAD_ITERATIONS ( (HTILE_LOCAL_SIZE + ROWS_PER_WORKGROUP-1) / ROWS_PER_WORKGROUP )

#if 0
void kernel test_kernel(__global const float* filter,
						__global const float* gradOutput,
						__global const float* weights,	
						__global const float* dataBuf3,
						__global const float* dataBuf4,
						__global const float* dataBuf5,
						__global float* gradInput,
						const int batchSize,  
						const int const2,  
						const int const3,  
						const int const4,  
						const int const5,  
						const int const6)  
#else 
void kernel calcGradInput_TileMode(
const int batchSize,
global const float *gradOutput, global const float *weights, global float *gradInput) 
#endif 
{
		int localId = get_local_id(0);
		int groupId = get_group_id(0);

		const int upstreamImage2dId = groupId / (HTILES * VTILES) ;
	  int n  						= upstreamImage2dId / gInputPlanes;
    int upstreamPlane = upstreamImage2dId % gInputPlanes;
	
		
	
		float sum = 0;	
	  float sum2 = 0;
		float sum3 = 0;
	  float sum4 = 0;
		__local float __gradOutput[HTILE_LOCAL_SIZE * VTILE_LOCAL_SIZE];
	
	
	  int weightPlaneOffset = upstreamPlane * gInputPlanes*gFilterSize*gFilterSize;
		int outputImageOffset = n * gNumFilters *  (gOutputSize *  gOutputSize);

    //Step1: calculuating output

		
		const int tile_BL_Y = ((groupId % (HTILES * VTILES)) / HTILES) * (VTILE_REPEAT * TILE_HEIGHT);
		const int  tile_BL_X = ((groupId % (HTILES * VTILES)) % HTILES) * TILE_WIDTH;

		int localRow  = localId / TILE_WIDTH ;
		int localCol  = localId % TILE_WIDTH ;
		
		int upstreamRow = localRow  + tile_BL_Y + gMargin; 
		int upstreamCol	= localCol  + tile_BL_X + gMargin;	
			
		bool bValid =  upstreamRow < gInputSize && upstreamCol < gInputSize && localId < MAX_VALID_ID;
		
		//2nd pixel vaid or not
		bool bValid2 =  (upstreamRow + TILE_HEIGHT*1) < gInputSize && upstreamCol < gInputSize && localId < MAX_VALID_ID;
		bool bValid3 =  (upstreamRow + TILE_HEIGHT*2) < gInputSize && upstreamCol < gInputSize && localId < MAX_VALID_ID;
		bool bValid4 =  (upstreamRow + TILE_HEIGHT*3) < gInputSize && upstreamCol < gInputSize && localId < MAX_VALID_ID;
		
		
		
		//Step2: loading_offset;
    int local_x, local_y;
		bool bImageLoad = false;
		bool bImageLoad2 = false;
		int image_x, image_y;
		if( localId < (ROWS_PER_WORKGROUP * HTILE_LOCAL_SIZE))
		{
			  bImageLoad = true;
				local_x	= localId % HTILE_LOCAL_SIZE;
			  local_y = localId / HTILE_LOCAL_SIZE;
			  
			  
			  image_x = local_x + tile_BL_X - (gFilterSize-1) + gMargin;
			  image_y = local_y + tile_BL_Y - (gFilterSize-1) + gMargin;
				if( image_x >= 0 && image_x < gInputSize)
				{
						bImageLoad2 = true;
				}
		}	

	  
		// aggregate over [outPlane][outRow][outCol]				
		int forceloop = min(gNumFilters, gNumFilters*batchSize);
		
    for (int outPlane = 0; outPlane < forceloop; outPlane++) {
			  //only bImageLoad == true will load images
			  //For exmaple 5x5 filters, 16x16  tile
		    //  1st load 12 Rows   == 256 threads == 256/ (16 + 5 -1 )  = 12:  
				//	 					==> 240 threds load iamge ,   last 16 threads do not load 
	      //  2nd load 4  rows   == 16- 12 = rows  
				//            ==> only 1st 64 threads load images  
        //  
			  
			   if(bImageLoad)
				 {
						//1st Load is a must 
					 {
								int localOffset =	local_y * HTILE_LOCAL_SIZE + local_x;
								float value;				
								int row = 0;
								bool bProcess = false;

								row = image_y; 						
								
								
								
								bProcess = row >= 0 && row < gInputSize && bImageLoad2 ;
								
								int offset =  outputImageOffset + 
																outPlane * gOutputSize * gOutputSize +
																row * gOutputSize + 
																image_x;
								
								 value =  gradOutput[offset];
								 if(!bProcess){
										value = 0;
									}
								
								__gradOutput[localOffset] = value;
						}			
						
						for(int i = 1; i < IMAGE_LOAD_ITERATIONS; i++)
						{
							int localOffset =	(local_y + i * ROWS_PER_WORKGROUP)  * HTILE_LOCAL_SIZE + local_x;
							
							//Boundary check
							if( localOffset < (HTILE_LOCAL_SIZE * VTILE_LOCAL_SIZE))
							{							  
									float value;				
									int row = 0;
									bool bProcess = false;

									row = image_y + i * ROWS_PER_WORKGROUP; 						
									
									
									
									bProcess = row >= 0 && row < gInputSize && bImageLoad2 ;
									
									int offset =  outputImageOffset + 
																	outPlane * gOutputSize * gOutputSize +
																	row * gOutputSize + 
																	image_x;
									
									 value =  gradOutput[offset];
									 if(!bProcess){
											value = 0;
										}
									
									__gradOutput[localOffset] = value;
							}			
						}		
				 }
				

				barrier(CLK_LOCAL_MEM_FENCE);
#if (gFilterSize > 5)				
				if(bValid){
					//No negative offset mode, it is slow code 					
					
					//LLVM with Pointers to produce less address
					__global const float *f = &weights[weightPlaneOffset + outPlane * gFilterSize *gFilterSize];
					for(int i=0; i < gFilterSize; i++)
					{
						for(int j=0; j < gFilterSize; j++)					
						{
								//shift local memory (gFilterSize-1) to image_xy
								int outRow = localRow - i + (gFilterSize-1);
								int outCol = localCol - j + (gFilterSize-1);
								
								int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;

								float thisError = __gradOutput[resultIndex];								
								float thisWeight = *(f + i * gFilterSize + j);
								sum += thisError* thisWeight;
						}
				

						//Reduce constant loading 
						if(bValid2)
						for(int j=0; j < gFilterSize; j++)						
						{
									//shift local memory (gFilterSize-1) to image_xy
								int outRow = localRow - i + (gFilterSize-1) + TILE_HEIGHT ;
								int outCol = localCol - j + (gFilterSize-1);
									
									int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;

									float thisError = __gradOutput[resultIndex];
									float thisWeight = *(f + i * gFilterSize + j);
									sum2 += thisError* thisWeight;
						}
						if(bValid3)
						for(int j=0; j < gFilterSize; j++)						
						{
									//shift local memory (gFilterSize-1) to image_xy
									int outRow = localRow - i + (gFilterSize-1) + TILE_HEIGHT *2 ;
									int outCol = localCol - j + (gFilterSize-1);
									
									int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;

									float thisError = __gradOutput[resultIndex];
									float thisWeight = *(f + i * gFilterSize + j);
									sum3 += thisError* thisWeight;
						}
						if(bValid4)
						for(int j=0; j < gFilterSize; j++)						
						{
									//shift local memory (gFilterSize-1) to image_xy
									int outRow = localRow - i + (gFilterSize-1) + TILE_HEIGHT *3 ;
									int outCol = localCol - j + (gFilterSize-1);
									
									int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;

									float thisError = __gradOutput[resultIndex];
									float thisWeight = *(f + i * gFilterSize + j);
									sum4 += thisError* thisWeight;
						}	
					}
				}
#else
				if(bValid){
					//No negative offset mode, it is slow code 					
						for(int i=0; i < gFilterSize; i++)
						{
							for(int j=0; j < gFilterSize; j++)					
							{
									//shift local memory (gFilterSize-1) to image_xy
									int outRow = localRow - i + (gFilterSize-1);
									int outCol = localCol - j + (gFilterSize-1);
									
									int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;

									float thisError = __gradOutput[resultIndex];
									int thisWeightIndex = weightPlaneOffset + 
																				outPlane * gFilterSize *gFilterSize
																				+ i * gFilterSize
																				+ j;
									float thisWeight = weights[thisWeightIndex];
									sum += thisError* thisWeight;
							}
						}
						if(bValid2){
						for(int i=0; i < gFilterSize; i++)
							for(int j=0; j < gFilterSize; j++)					
								{
								
										//shift local memory (gFilterSize-1) to image_xy
										int outRow = localRow - i + (gFilterSize-1) + TILE_HEIGHT;
										int outCol = localCol - j + (gFilterSize-1);
										
										int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;

										float thisError = __gradOutput[resultIndex];
									int thisWeightIndex = weightPlaneOffset + 
																				outPlane * gFilterSize *gFilterSize
																				+ i * gFilterSize
																				+ j;
										float thisWeight = weights[thisWeightIndex];
										sum2 += thisError* thisWeight;
								}
						}				
						
						if(bValid3){
						for(int i=0; i < gFilterSize; i++)
							for(int j=0; j < gFilterSize; j++)					
									{
										//shift local memory (gFilterSize-1) to image_xy
										int outRow = localRow - i + (gFilterSize-1) + TILE_HEIGHT*2;
										int outCol = localCol - j + (gFilterSize-1);
										
										int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;

										float thisError = __gradOutput[resultIndex];
									int thisWeightIndex = weightPlaneOffset + 
																				outPlane * gFilterSize *gFilterSize
																				+ i * gFilterSize
																				+ j;
										float thisWeight = weights[thisWeightIndex];
										sum3 += thisError* thisWeight;
								}
						}			

						if(bValid4){
						for(short i=0; i < gFilterSize; i++)
							for(short j=0; j < gFilterSize; j++)
								{
										//shift local memory (gFilterSize-1) to image_xy
										int outRow = localRow - i + (gFilterSize-1) + TILE_HEIGHT*3;
										int outCol = localCol - j + (gFilterSize-1);
										
										int resultIndex = outRow * HTILE_LOCAL_SIZE + outCol;

										float thisError = __gradOutput[resultIndex];
										int thisWeightIndex = weightPlaneOffset + 
																				outPlane * gFilterSize *gFilterSize
																				+ i * gFilterSize
																				+ j;
										float thisWeight = weights[thisWeightIndex];
										sum4 += thisError* thisWeight;
								}
						}							
				}
#endif					
				
				
    }//while
		
		if (bValid) 
		{

		   unsigned	int Offset  = upstreamImage2dId * gInputSizeSquared + upstreamRow * gInputSize + upstreamCol;
			
        gradInput[Offset] = sum;
			 if(bValid2)
			 {
				 gradInput[Offset + TILE_HEIGHT * gInputSize] = sum2;
			 }
			 if(bValid3)
			 {
				 gradInput[Offset + TILE_HEIGHT * 2 * gInputSize] = sum3;
			 }
			 if(bValid4)
			 {
				 gradInput[Offset + TILE_HEIGHT * 3 * gInputSize] = sum4;
			 }
			 
    }

}

#if 0
void kernel calcGradInput( 
        const int batchSize,
        global const float *gradOutput, global float *weights, global float *gradInput) {
    int globalId = get_global_id(0);

    const int upstreamImage2dId = globalId / gInputSizeSquared;

    const int intraImageOffset = globalId % gInputSizeSquared;
    const int upstreamRow = intraImageOffset / gInputSize;
    const int upstreamCol = intraImageOffset % gInputSize;

    const int upstreamPlane = upstreamImage2dId % gInputPlanes;
    const int n = upstreamImage2dId / gInputPlanes;

    if (n >= batchSize) {
        return;
    }

    const int minFilterRow = max(0, upstreamRow + gMargin - (gOutputSize - 1));
    const int maxFilterRow = min(gFilterSize - 1, upstreamRow + gMargin);
    const int minFilterCol = max(0, upstreamCol + gMargin - (gOutputSize -1));
    const int maxFilterCol = min(gFilterSize - 1, upstreamCol + gMargin);

    float sumWeightTimesOutError = 0;
    // aggregate over [outPlane][outRow][outCol]
    for (int outPlane = 0; outPlane < gNumFilters; outPlane++) {
        for (int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++) {
            int outRow = upstreamRow + gMargin - filterRow;
            for (int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++) {
                int outCol = upstreamCol + gMargin - filterCol;
                int resultIndex = (( n * gNumFilters 
                          + outPlane) * gOutputSize
                          + outRow) * gOutputSize
                          + outCol;
                float thisError = gradOutput[resultIndex];
                int thisWeightIndex = (( outPlane * gInputPlanes
                                    + upstreamPlane) * gFilterSize
                                    + filterRow) * gFilterSize
                                    + filterCol;
                float thisWeight = weights[thisWeightIndex];
                float thisWeightTimesError = thisWeight * thisError;
                sumWeightTimesOutError += thisWeightTimesError;
            }
        }
    }
    gradInput[globalId] = sumWeightTimesOutError;
}

CPU source code 
    for(int n = 0; n < batchSize; n++) {
        for(int upstreamPlane = 0; upstreamPlane < dim.inputPlanes; upstreamPlane++) {
            for(int upstreamRow = 0; upstreamRow < dim.inputSize; upstreamRow++) {
                int minFilterRow = std::max(0, upstreamRow + margin - (dim.outputSize - 1));
                int maxFilterRow = std::min(dim.filterSize - 1, upstreamRow + margin);
                for(int upstreamCol = 0; upstreamCol < dim.inputSize; upstreamCol++) {
                    float sumWeightTimesGradOutput = 0;
                    // aggregate over [outPlane][outRow][outCol]
                    int minFilterCol = std::max(0, upstreamCol + margin - (dim.outputSize -1));
                    int maxFilterCol = std::min(dim.filterSize - 1, upstreamCol + margin);
                    for(int outPlane = 0; outPlane < dim.numFilters; outPlane++) {
                        for(int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++) {
                            int outRow = upstreamRow + margin - filterRow;
                            for(int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++) {
                                int outCol = upstreamCol + margin - filterCol;
                                int resultIndex = (( n 
                                    * dim.numFilters + outPlane)
                                    * dim.outputSize + outRow)
                                    * dim.outputSize + outCol;
                                float thisGradOutput = gradOutput[resultIndex];
                                int thisWeightIndex = (( outPlane 
                                    * dim.inputPlanes + upstreamPlane)
                                    * dim.filterSize + filterRow)
                                    * dim.filterSize + filterCol;
                                float thisWeight = weights[thisWeightIndex];
                                sumWeightTimesGradOutput += thisWeight * thisGradOutput;
                            }
                        }
                    }
                    int inputIndex = (( n
                        * dim.inputPlanes + upstreamPlane)
                        * dim.inputSize + upstreamRow)
                        * dim.inputSize + upstreamCol;
                    gradInput[inputIndex] = sumWeightTimesGradOutput; // * activationDerivativeUpstream;
                }
            }
        }
    }


#endif