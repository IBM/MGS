// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-12-30-2018
//
// (C) Copyright IBM Corp. 2005-2018  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>

/* helper utilities */
template<class StateType, class DerivedType = StateType>
void resize(const StateType &in, DerivedType &out) 
{
  // standard implementation works for containers
  out.resize(in.size());
};
template<class DerivedType>
void resize(const size_t &N, DerivedType&out) 
{
  // standard implementation works for containers
  out.resize(N);
};

// specialization for those that cannot be resized
// e.g. boost::array
//  or those use a different method for resizing
//template<class T, size_t N>
//void resize(const boost::array<T, N> &, boost::array<T,N>& ) {
//  /* boost::arrays live on the stack */
//}

/* abstract representation for the loop */
struct container_algebra
{
  template<class S1, class S2, class S3, class Op>
    static void for_each3(S1 &s1, S2 &s2, S3 &s3, Op op)
    {
      /* s3[i] = op(s1[i], s2[i])
      /* can be generalized to use iterators on containers S1, S2, S3 */
      const size_t dim = s1.size();
      for (size_t n = 0; n < dim; ++n)
      {
        op(s1[n], s2[n], s3[n]);
      }
    }
  template<class S1, class S2, class S3, class S4, class S5, class S6, class Op>
    static void for_each6(S1 &s1, S2 &s2, S3 &s3, S4 &s4, S5 &s5, S6 &s6, Op op)
    {
      /* can be generalized to use iterators on containers S1, S2, S3 */
      const size_t dim = s1.size();
      for (size_t n = 0; n < dim; ++n)
      {
        op(s1[n], s2[n], s3[n], s4[n], s5[n], s6[n]);
      }
    }
};

/* abstract representation for the operation on each element 
 *  within the loop using 2 functor types */
struct default_operations 
{
  template<class Fac1 = double, class Fac2 = Fac1>
    struct scale_sum2{
      typedef void result_type;
      const Fac1 alpha1; 
      const Fac2 alpha2;
      scale_sum2(Fac1 alpha1, Fac2 alpha2): alpha1(alpha1), alpha2(alpha2){}
      template<class T0, class T1, class T2>
        void operator() (T0 &t0, const T1 &t1, const T2 & t2) const {
          t0 = alpha1 * t1 + alpha2 * t2;
        }
    };
  template<class Fac1 = double, class Fac2 = Fac1,
    class Fac3 = Fac2, class Fac4 = Fac2, class Fac5 = Fac2
    >
    struct scale_sum5{
      typedef void result_type;
      const Fac1 alpha1; 
      const Fac2 alpha2;
      const Fac3 alpha3; 
      const Fac4 alpha4;
      const Fac5 alpha5;
      scale_sum5(Fac1 alpha1, Fac2 alpha2, Fac3 alpha3, Fac4 alpha4, Fac5 alpha5): alpha1(alpha1), alpha2(alpha2), alpha3(alpha3), alpha4(alpha4), alpha5(alpha5){}
      template<class T0, class T1, class T2, class T3, class T4, class T5>
        void operator() (T0 &t0, const T1 &t1, const T2 & t2, const T3 &t3, const T4 &t4, const T5 &t5) const {
          t0 = alpha1 * t1 + alpha2 * t2 + alpha3 * t3 + alpha4 * t4 + alpha5 * t5;
        }
    };
};

struct thrust_algebra 
{
  template<class S1, class S2, class S3, class Op> 
    static 
    void for_each3(S1 &s1, S2 &s2, S3 &s3, Op op) {
      thrust::for_each(
          /* create the tuple, which we have to unpack later 
           *
           * To access the individual elements, we have to unpack the tuple, 
           * which can be done by the Thrust’s get<N>(tuple) function 
           * that simply returns the N-th entry of the given tuple
           * */
          thrust::make_zip_iterator( thrust::make_tuple(
              s1.begin(), s2.begin(), s3.begin() ) ),
          thrust::make_zip_iterator( thrust::make_tuple(
              s1.end(), s2.end(), s3.end() ) ),
          op);

  }
};
struct thrust_operations 
{
  template<class Fac1 = double, class Fac2 = Fac1> 
    struct scale_sum2 {
      const Fac1 m_alpha1;
      const Fac2 m_alpha2;
      scale_sum2(const Fac1 alpha1, const Fac2 alpha2)
        : m_alpha1(alpha1), m_alpha2(alpha2) { } 
      template< class Tuple >
        __host__ __device__ void operator()(Tuple t) const {
          thrust::get<0>(t) = m_alpha1 * thrust::get<1>(t) +
            m_alpha2 * thrust::get<2>(t);
        } 
    };
};

/* 
 * StateType = a system of 'N' variables, each of type VarType
 *     this can be std::vector< VarType >
 *     or boost::array< VarType, N >
 *     or thrust::device_vector< VarType >
 *     or ShallowArray_Flat< VarType, Array_Flat<int>::MemLocation::UNIFIED_MEM >
 *     or ShallowArray_Flat< VarType, Array_Flat<int>::MemLocation::CPU >
 * DerivedType = a system of 'N' variables representing dx/dt 
 * VarType = type of each variable
 * TimeType = type of the time data
 * Algebra = the functor that provide operators on containers, e.g. for_loop3
 * Operatons = the functor that provide operators on elements, presumbly same index, in containers
 */
template<class StateType,
  class DerivedType = StateType,
  class VarType = double,
  class TimeType = VarType,
  class Algebra = container_algebra,
  class Operations = default_operations
  >
  class NumSolver_RK4
{
  public:
  NumSolver_RK4(): N(0){}
  /* System refers to the system function which can be
   *    function pointers
   *    function object
   *    generalized functions objects (std::function, boost::function)
   *    C++11 lambdas
   * as long as it defines a function call operator that accepts 3 arguments
   * System{
   *   operator()(StateType &x, StateType &dxdt, TimeType t)
   * }
   */
  template <typename System>
    void do_step(System* system, StateType &x, TimeType t, TimeType dt)
    {
      const TimeType dt2 = dt/2;
      const TimeType dt3 = dt/3;
      const TimeType dt6 = dt/6;
      if (N < x.size())
        adjust_size(x);
      assert(N = x.size());
      typedef typename Operations::template scale_sum2<VarType, TimeType> scale_sum2;
      typedef typename Operations::template scale_sum5<VarType, TimeType, TimeType, TimeType, TimeType> scale_sum5;

      (*system)(x, k1, t);// k1 = RHS(x,t) use x to calculate k1
      /*
      for(size_t i = 0; i < N; ++i)
        x_tmp[i] = x[i] + dt2 * k1[i];
        */
      const VarType one = 1;
      /* x_tmp[i] = one * x[i] + dt2 * k1[i] */
      Algebra::for_each3(x_tmp, x, k1, scale_sum2(one, dt2));
      (*system)(x_tmp, k2, t + dt2);
      /*
      for(size_t i = 0 ; i < N; ++i)
        x_tmp[i] = x[i] + dt2 * k2[i];
        */
      Algebra::for_each3(x_tmp, x, k2, scale_sum2(one, dt2));
      (*system)(x_tmp, k3, t + dt2);
      /*
      for(size_t i = 0; i < N; ++i)
        x_tmp[i] = x[i] + dt * k3[i];
        */
      Algebra::for_each3(x_tmp, x, k3, scale_sum2(one, dt));
      (*system)(x_tmp, k4, t + dt);
      /*
      for(size_t i = 0; i < N; ++i)
        x[i] = x[i] + dt6*k1[i] + dt3*k2[i] + dt3*k3[i] + dt6*k4[i];
        */
      Algebra::for_each6(x, x, k1, k2, k3, k4, scale_sum5(one, dt6, dt3, dt3, dt6));
    }
    inline TimeType get_dT(const TimeType& dt, const int& ii)
    {
      const TimeType dt2 = dt/2;
      const TimeType dt3 = dt/3;
      const TimeType dt6 = dt/6;
      switch (ii)
      {
        case 1: return 0;
        case 2: return dt2;
        case 3: return 0;
        case 4: return dt-dt2;
        default: return 0;
      }
    }
    template <typename System>
    void do_step_small(System* system, StateType &x, TimeType t, TimeType dt, int ii)
    {
      const TimeType dt2 = dt/2;
      const TimeType dt3 = dt/3;
      const TimeType dt6 = dt/6;
      if (ii == 1 and N < x.size())
        adjust_size(x);
      assert(N = x.size());
      typedef typename Operations::template scale_sum2<VarType, TimeType> scale_sum2;
      typedef typename Operations::template scale_sum5<VarType, TimeType, TimeType, TimeType, TimeType> scale_sum5;

      const VarType one = 1;
      if (ii == 1)
      {
        (*system)(x, k1, t);//use x to calculate k1
        /*
           for(size_t i = 0; i < N; ++i)
           x_tmp[i] = x[i] + dt2 * k1[i];
           */
        Algebra::for_each3(x_tmp, x, k1, scale_sum2(one, dt2));
      }
      if (ii == 2)
      {
        (*system)(x_tmp, k2, t + dt2);
        /*
           for(size_t i = 0 ; i < N; ++i)
           x_tmp[i] = x[i] + dt2 * k2[i];
           */
        Algebra::for_each3(x_tmp, x, k2, scale_sum2(one, dt2));
      }
      if (ii == 3)
      {
        (*system)(x_tmp, k3, t + dt2);
        /*
           for(size_t i = 0; i < N; ++i)
           x_tmp[i] = x[i] + dt * k3[i];
           */
        Algebra::for_each3(x_tmp, x, k3, scale_sum2(one, dt));
      }
      if (ii == 4)
      {
        (*system)(x_tmp, k4, t + dt);
        /*
           for(size_t i = 0; i < N; ++i)
           x[i] += dt6*k1[i] + dt3*k2[i] + dt3*k3[i] + dt6*k4[i];
           */
        Algebra::for_each6(x, x, k1, k2, k3, k4, scale_sum5(one, dt6, dt3, dt3, dt6));
      }
      if (ii > 4)
        assert(0);
    }
  private:
  size_t N; /* how many unknown time-dependent variables */
  StateType x_tmp;
  //StateType x;
  /*
   * derivatives might require a representation different from the state
   * especially if arithmetic types with dimensions are used, for example the ones from Boost.Unit
   */
  DerivedType k1, k2, k3, k4;
  protected:
  /* adjust size of the system using the size value N */
  void adjust_size(const size_t& _N)
  {
    N = _N;
    resize(N, x_tmp);
    resize(N, k1);
    resize(N, k2);
    resize(N, k3);
    resize(N, k4);
  };
  /* adjust size of the system using information from another StateType */
  void adjust_size(const StateType &x)
  {
    resize(x, x_tmp);
    resize(x, k1);
    resize(x, k2);
    resize(x, k3);
    resize(x, k4);
    N = x.size();
  };

};
typedef NumSolver_RK4< std::vector<double> > RK4_CPU_stepper_double;
typedef NumSolver_RK4< std::vector<float> > RK4_CPU_stepper_float;
//typedef NumSolver_RK4< ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>,
//       GpuAlgebra, GpuOperations > RK4_GPU;

