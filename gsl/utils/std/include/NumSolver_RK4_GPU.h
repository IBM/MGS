
#ifndef NUMSOLVER_RK4_GPU_H
#define NUMSOLVER_RK4_GPU_H

#include "NumSolver_RK4.h"

template<class S0, class S1, class S2, class VarType, class TimeType>
void __global__ RK4_for_each3(S0* s0, S1* s1, S2* s2, VarType a1, TimeType a2, size_t size)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
      s0[index] = a1* s1[index] + a2 * s2[index];
   }
}

template<class S0, class S1, class S2, class S3, class S4, class S5, class VarType, class TimeType>
void __global__ RK4_for_each6(S0 *s0, S1 *s1, S2 *s2, S3 *s3, S4 *s4, S5 *s5, 
   VarType a1, TimeType a2, TimeType a3, TimeType a4, TimeType a5, size_t size)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
      s0[index] = a1 * s1[index] + a2 * s2[index] + a3 * s3[index] + a4 * s4[index] + a5 * s5[index];
   }
}

struct container_algebra_gpu
{
    template <typename... Arguments>
    static void for_each3(int blocks, int threads, 
         //StateType &x, TimeType t, TimeType dt, int ii,
         void (*f)(Arguments...), Arguments... args)
    {
      RK4_for_each3<<<blocks, threads>>>(args...);
    }
    template <typename... Arguments>
    static void for_each6(int blocks, int threads, 
         //StateType &x, TimeType t, TimeType dt, int ii,
         void (*f)(Arguments...), Arguments... args)
    {
      RK4_for_each6<<<blocks, threads>>>(args...);
    }
};
//typedef NumSolver_RK4< ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>,
//       GpuAlgebra, GpuOperations > RK4_GPU;

template <
    typename SolverType,
    typename StateType,
    typename DerivedType = StateType,
    typename VarType = double,
    typename TimeType = VarType,
    typename Algebra = container_algebra_gpu,
    typename... Arguments>
void do_step_small_on_gpu(int blocks, int threads, 
      SolverType &solver,
      StateType &x, TimeType t, TimeType dt, int ii,
      void (*f)(Arguments...), Arguments... args)
{
  //size_t total_elements, [> P*stride <]
  size_t total_elements = solver.get_total_element();
  const TimeType dt2 = dt/2;
  const TimeType dt3 = dt/3;
  const TimeType dt6 = dt/6;
  //if (ii == 1 and N < x.size())
  //  adjust_size(x);
  //assert(N = x.size());

  const VarType one = 1;
  if (ii == 1)
  {
    f<<<blocks, threads>>>(args...);
    //(*system)(x, k1, t);//use x to calculate k1

    /*
        for(size_t i = 0; i < N; ++i)
        x_tmp[i] = x[i] + dt2 * k1[i];
        */
    //Algebra::for_each3(x_tmp, x, k1, scale_sum2(one, dt2), blocks, threads);
    StateType&x_tmp = solver.get_state_tmp();
    DerivedType &k1 = solver.get_k(ii);
    int THREADS_PER_BLOCK= 256;
    int BLOCKS= ceil((float)total_elements/ THREADS_PER_BLOCK);
    Algebra::for_each3(
      //blocks, threads,
      BLOCKS, THREADS_PER_BLOCK,
      RK4_for_each3,
      x_tmp.getDataRef(), x.getDataRef(), k1.getDataRef(), 
      one, dt2,
      total_elements);
  }
  if (ii == 2)
  {
    f<<<blocks, threads>>>(args...);
    /*
        for(size_t i = 0 ; i < N; ++i)
        x_tmp[i] = x[i] + dt2 * k2[i];
        */
    //Algebra::for_each3(x_tmp, x, k2, scale_sum2(one, dt2));
    StateType&x_tmp = solver.get_state_tmp();
    DerivedType &k2 = solver.get_k(ii);
    int THREADS_PER_BLOCK= 256;
    int BLOCKS= ceil((float)total_elements/ THREADS_PER_BLOCK);
    Algebra::for_each3(
      //blocks, threads,
      BLOCKS, THREADS_PER_BLOCK,
      RK4_for_each3,
      x_tmp.getDataRef(), x.getDataRef(), k2.getDataRef(), 
      one, dt2,
      total_elements);
  }
  if (ii == 3)
  {
    f<<<blocks, threads>>>(args...);
    /*
        for(size_t i = 0; i < N; ++i)
        x_tmp[i] = x[i] + dt * k3[i];
        */
    //Algebra::for_each3(x_tmp, x, k3, scale_sum2(one, dt));
    StateType&x_tmp = solver.get_state_tmp();
    DerivedType &k3 = solver.get_k(ii);
    int THREADS_PER_BLOCK= 256;
    int BLOCKS= ceil((float)total_elements/ THREADS_PER_BLOCK);
    Algebra::for_each3(
      //blocks, threads,
      BLOCKS, THREADS_PER_BLOCK,
      RK4_for_each3,
      x_tmp.getDataRef(), x.getDataRef(), k3.getDataRef(), 
      one, dt,
      total_elements);
  }
  if (ii == 4)
  {
    f<<<blocks, threads>>>(args...);
    /*
        for(size_t i = 0; i < N; ++i)
        x[i] += dt6*k1[i] + dt3*k2[i] + dt3*k3[i] + dt6*k4[i];
        */
    //Algebra::for_each6(x, x, k1, k2, k3, k4, scale_sum5(one, dt6, dt3, dt3, dt6));
    StateType&x_tmp = solver.get_state_tmp();
    DerivedType &k1 = solver.get_k(1);
    DerivedType &k2 = solver.get_k(2);
    DerivedType &k3 = solver.get_k(3);
    DerivedType &k4 = solver.get_k(4);
    int THREADS_PER_BLOCK= 256;
    int BLOCKS= ceil((float)total_elements/ THREADS_PER_BLOCK);
    Algebra::for_each6(
      //blocks, threads,
      BLOCKS, THREADS_PER_BLOCK,
      RK4_for_each6,
      x.getDataRef(), x.getDataRef(), 
      k1.getDataRef(), k2.getDataRef(), 
      k3.getDataRef(), k4.getDataRef(), 
      one, dt6, dt3, dt3, dt6,
      total_elements);
  }
  if (ii > 4)
    assert(0);
}
#endif
