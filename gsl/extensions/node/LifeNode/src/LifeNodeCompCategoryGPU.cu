
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define GPU_OPTION1
#if defined(HAVE_GPU) || defined(__CUDACC__)
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#if defined(HAVE_GPU) and defined(GPU_OPTION1)
struct _Data_LifeNode {
   thrust::device_vector<int> value;
   thrust::device_vector<int> publicValue;
   thrust::device_vector<int*> neighbors;
} Data_LifeNode;

namespace LifeNodeCompCategory
{
CUDA_CALLABLE void	update(RNG& rng)
{
	double			CUDA_result[BLOCKS_ANGLE], angle;
	int			CUDA_tid[BLOCKS_ANGLE], index;
	Cartesian		GeoSat = findStartArcGeo(EarthStation.ToLLA());

	kernelAngle <<< BLOCKS_ANGLE, THREADS_ANGLE >>> (EarthStation, Sat, GeoSat, CUDA_result, CUDA_tid);
	angle = getMinusAngle(CUDA_result);
	index = getIndexPosition(CUDA_result, CUDA_tid);
	// printf //
}
void LifeNode::update(RNG& rng) 
{
   int neighborCount=0;
   ShallowArray<int*>::iterator iter, end = neighbors.end();
   for (iter=neighbors.begin(); iter!=end; ++iter) {
     neighborCount += **iter;
   }
   
   if (neighborCount<= getSharedMembers().tooSparse || neighborCount>=getSharedMembers().tooCrowded) {
     value=0;
   }
   else {
     value=1;
   }
}


}
#endif
