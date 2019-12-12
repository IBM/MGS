//Define a set of builtin math functions that can be used from GPU-side
#ifndef CUDA_ON_DEVICE_BUILTINS_H
#define CUDA_ON_DEVICE_BUILTINS_H

#if defined(HAVE_GPU) 
//referenced from https://github.com/eyalroz/libgiddy/blob/master/src/cuda/on_device/builtins.cuh
#define __df__ __device__ __forceinline__

template <typename T> __df__ T minimum(T x, T y);
template <> __df__ int                 minimum<int               >(int x, int y)                               { return (int)fmin((double)x,(double)y);    }
template <> __df__ size_t              minimum<size_t            >(size_t x, size_t y)                         { return (size_t)fmin((double)x,(double)y);    }
//template <> __df__ unsigned int        minimum<unsigned          >(unsigned int x, unsigned int y)             { return umin(x,y);   }
//template <> __df__ long                minimum<long              >(long x, long y)                             { return llmin(x,y);  }
//template <> __df__ unsigned long       minimum<unsigned long     >(unsigned long x, unsigned long y)           { return ullmin(x,y); }
//template <> __df__ long long           minimum< long long        >(long long x, long long y)                   { return llmin(x,y);  }
//template <> __df__ unsigned long long  minimum<unsigned long long>(unsigned long long x, unsigned long long y) { return ullmin(x,y); }
template <> __df__ float               minimum<float             >(float x, float y)                           { return fminf(x,y);  }
template <> __df__ double              minimum<double            >(double x, double y)                         { return fmin(x,y);   }

#endif
#endif
