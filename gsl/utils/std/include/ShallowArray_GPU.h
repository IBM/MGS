// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef SHALLOWARRAY_GPU_H
#define SHALLOWARRAY_GPU_H

#include <cstring>
#include <typeinfo>

#include <cxxabi.h>

template <class T>
T * new_memory_scalar(size_t size=1)
{//array of scalar
//IMPORTANT: bug here -> as if T is a struct, its data members as arrays are not configured so that 
//   memLocation is on Unified Memory
  //use this if T is not a ShallowArray_Flat 
   T* ptr; //void *ptr;
   int len = size* sizeof(T);
   gpuErrorCheck(cudaGetLastError());
   gpuErrorCheck(cudaMallocManaged(&ptr, len));
   gpuErrorCheck(cudaDeviceSynchronize());
   return ptr;
}
template <class T>
T * new_memory_array(size_t size=1)
{//array of array
   T* ptr; //void *ptr;
   int len = size* sizeof(T);
   gpuErrorCheck(cudaGetLastError());
   gpuErrorCheck(cudaMallocManaged(&ptr, len));
   for (auto ii=0; ii<size; ii++)
      ptr[ii].____set_mem(Array_Flat<int>::MemLocation::UNIFIED_MEM);
   gpuErrorCheck(cudaDeviceSynchronize());
   return ptr;
}
template <class T>
void delete_memory(T*& ptr)
{
   gpuErrorCheck(cudaDeviceSynchronize());
   gpuErrorCheck(cudaFree(ptr));         
}

inline std::string get_realname(const std::type_info& ti)
{
  char   *realname;
  int     status;
  realname = abi::__cxa_demangle(ti.name(), 0, 0, &status);
    //std::cout << ti.name() << "\t=> " << realname << "\t: " << status << '\n';
  std::string rname(realname);
  free(realname);
  return rname;
};

//#define USE_SIMPLE_CUDA_ARRAY

#if defined(USE_SIMPLE_CUDA_ARRAY)
/* 
IMPORTANT
If we don't want to make ShallowArray_Flat to be on Unified Memory, then 
  array of array should be ShallowArray_Flat< LocalVector<DataType>, Unified_Mem >
*/

//#define USE_OVERLOADED_NEW

typedef enum {CPU=0, UNIFIED_MEM, FPGA} MemLocation; //IMPORTANT: Keep CPU=0
typedef enum {DOUBLE_ARRAY, FLAT_ARRAY} ArrayDesign;
namespace details{
  namespace Allocator{
    template <typename T, int memLocation> __managed__ int _mem_location(memLocation);
  }
}

//#undef CUDA_CALLABLE
//#define CUDA_CALLABLE

template <typename T, int memLocation>
class Allocator
{
public:
    //TUAN: make this public so that we can use
    // Array_Flat<int>::MemLocation::CPU

    //static int _mem_location;

    //using cudaAllocator_tag = std::true_type;
    //using hostAllocator_tag = std::false_type;
    //using cudaAllocator_tag = typename std::integral_constant<int, MemLocation::UNIFIED_MEM>;
    //using hostAllocator_tag = typename std::integral_constant<int, MemLocation::CPU>;
    //using fpgaAllocator_tag = typename std::integral_constant<int, MemLocation::FPGA>;
    typedef std::integral_constant<int, MemLocation::UNIFIED_MEM> cudaAllocator_tag;
    typedef std::integral_constant<int, MemLocation::CPU>        hostAllocator_tag;
    typedef std::integral_constant<int, MemLocation::FPGA>       fpgaAllocator_tag;
    //using isCudaAllocator   = typename std::is_same<Alloc, cudaAllocator<value_type>>;
    CUDA_CALLABLE
    int mem_location() {
      return details::Allocator::_mem_location<T, memLocation>;
    }

    //static void *operator new(unsigned long int len)
    //{
    //  //return create(len, isCudaAllocator());
    //  std::cout << "new in Allocator\n";
    //  if (mem_location() == MemLocation::CPU)
    //    return create(len, hostAllocator_tag{});
    //  else if (mem_location() == MemLocation::UNIFIED_MEM)
    //    return create(len, cudaAllocator_tag{});
    //  else 
    //    assert(0);
    //}

    //static void operator delete(void *ptr) 
    //{
    //  std::cout << "delete in Allocator\n";
    //    //destroy(ptr, isCudaAllocator());
    //    if (mem_location() == MemLocation::CPU)
    //      destroy(ptr, hostAllocator_tag{});
    //    else if (mem_location() == MemLocation::UNIFIED_MEM)
    //      destroy(ptr, cudaAllocator_tag{});
    //    else 
    //      assert(0);
    //}

protected:
    static inline void *create(size_t len, cudaAllocator_tag)
    {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }

    static inline void destroy(void *ptr, cudaAllocator_tag)
    {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }

    static inline void *create(size_t len, hostAllocator_tag)
    {
        return ::new T(len);
    }

    static inline void destroy(void *ptr, hostAllocator_tag)
    {
        ::delete(static_cast<T*>(ptr));
    }
};

//template <typename T, int memLocation> 
//int Allocator<T, memLocation>::_mem_location(memLocation);

//template<class T>
//using Managed = Allocator<T, Allocator<int,0>::UNIFIED_MEM>; // type-id is vector<T, int>>

template <typename T, int memLocation=0, unsigned blockIncrementSize = SUGGESTEDBLOCKINCREMENTSIZE>
class ShallowArray_Flat 
//#ifdef USE_OVERLOADED_NEW
//: public Allocator<T, memLocation>
//#endif
{
private:
    T* m_begin;
    T* m_end;
   
    uint8_t _array_design;

    unsigned long int _allocated_size;
    unsigned long int length;
    int _mem_location;
    unsigned _incremental_size;  // the extra #elements to be allocated
    // NOTE: Arrays are organized in the form of multiple 'logical blocks'
    //       i.e. memory increase/reduced, in the form of one or many blocks
    unsigned _communicatedSize; //the number of elements MPI has communicated
    unsigned _sizeToCommunicate; //the number of elements MPI is to communicate
#if ! defined(__CUDA_ARCH__)
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_STD_ALLOCATOR
    std::allocator<T> allocator;
#endif
#endif
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_PLACEMENT_NEW
    char* pBuffer;
#endif
    CUDA_CALLABLE
    int mem_location() {
      return _mem_location;
    }
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_PLACEMENT_NEW
    CUDA_CALLABLE T* new_memory(unsigned long int len, char** new_pBuffer)
    {
      T* ptr=0; //void *ptr;
      {
        // https://stackoverflow.com/questions/15254/can-placement-new-for-arrays-be-used-in-a-portable-way 
        size_t NUMELEMENTS = len/sizeof(T);
        //char* pBuffer;
        //pBuffer = malloc(len);
        //pBuffer = new char[len];
        *new_pBuffer = new char[len];
        /* placement new on array */
        //ptr = new (pBuffer) T[len/sizeof(T)];
        /* placement new on individual element */
        /* we can use multiple thread here as well? */
        //ptr = (T*)pBuffer;
#if defined(DEBUG_SIZE_INFO)
        if (sizeGB >= 1)
        {
          std::cout << "    pass new char" << "\n";
        }
#endif
        //
        ptr = (T*)*new_pBuffer;
        /* serisal version */
        //https://stackoverflow.com/questions/4011577/placement-new-array-alignment?rq=1
        //https://stackoverflow.com/questions/4754763/object-array-initialization-without-default-constructor

        for(size_t i = 0; i < NUMELEMENTS; ++i)
        {
          //&ptr[i] = new (ptr + i) T();
          new (ptr + i) T();
#if defined(DEBUG_SIZE_INFO)
          if (sizeGB >= 1)
          {
            if (i > 0 and i % 100000000 == 0)
              std::cout << "    index "<< i << "\n";
          }
#endif
        }
        //end serial version

        /* parallel version */
        //       const size_t nthreads = std::min(10, (int)std::thread::hardware_concurrency());
        //       {
        //	  // Pre loop
        //	  //std::cout<<"parallel ("<<nthreads<<" threads):"<<std::endl;
        //	  std::vector<std::thread> threads(nthreads);
        //	  std::mutex critical;
        //	  const size_t nloop = NUMELEMENTS; 
        //	  for(int t = 0;t<nthreads;t++)
        //	  {
        //	     /* each thread bind to a lambdas function
        //		   the lambdas function accepts 3 arguments: 
        //		      t= thread index
        //		      bi = start index of data
        //		      ei = end index of data
        //		*/
        //	     threads[t] = std::thread(std::bind(
        //	     [&](const size_t bi, const size_t ei, const int t)
        //	     {
        //		// loop over all items
        //		for(size_t gn = bi; gn <ei; gn++)
        //		{
        //		  new (ptr + gn) T();
        //		}
        //		{
        //		//critical region
        //		std::lock_guard<std::mutex> lock(critical);
        //
        //#if defined(DEBUG_SIZE_INFO)
        //	if (sizeGB >= 1)
        //	{
        //	    std::cout << "    done thread "<< t << "\n";
        //	}
        //#endif
        //		}
        //	     }, t*nloop/nthreads, (t+1)==nthreads?nloop:(t+1)*nloop/nthreads, t));
        //	  }
        //	  std::for_each(threads.begin(),threads.end(),[](std::thread& x){x.join();});
        //	  // Post loop
        //	  // ..nothing
        ////#ifdef DEBUG_TIMER 
        ////       sim->benchmark_timelapsed_diff("... time for multithread() 20threads" );
        ////#endif
        //       }
      }
      return ptr;
    }
#endif
    CUDA_CALLABLE T * new_memory(unsigned long int len)
    {
      T* ptr=0; //void *ptr;
      //if (Allocator<T, memLocation>::mem_location()== MemLocation::CPU)
      if (mem_location()== MemLocation::CPU)
      {
#if ! defined(__CUDA_ARCH__)
        std::cout << "new_memory CPU ShallowArray_Flat: "
          << get_realname(typeid(T)) << '\n'; 
        //#define DEBUG_SIZE_INFO
#if defined(DEBUG_SIZE_INFO)
        double sizeGB = ((double)len) /1024/1024/1024;
        double sizeMB = ((double)len) /1024/1024;
        if (sizeGB >= 1)
        {
          std::cout << "AAA" << sizeGB << "GB for " 
            << get_realname(typeid(T)) << "\n";
        }
        else if (sizeMB >= 1)
        {
          std::cout << "BBB" << sizeMB << "MB for " 
            << get_realname(typeid(T)) << "\n";
        }
#endif
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_STD_ALLOCATOR
        {
          ptr = allocator.allocate(len/sizeof(T));
        }
#else
        {
          //ptr = ::new T[len/sizeof(T)];
          //ptr = ::new T[len/sizeof(T)];
          ptr = new T[len/sizeof(T)];
          //size_t numObjs = len/sizeof(T);
          //ptr = new T[numObjs];
        }
#endif
#if defined(DEBUG_SIZE_INFO)
        if (sizeGB >= 1)
        {
          std::cout << "end" << sizeGB << "GB\n";
        }
        else if (sizeMB >= 1)
        {
          std::cout << "end" << sizeMB << "MB\n";
        }
#endif
#else
        assert(0);
#endif
      }
      //else if (Allocator<T, memLocation>::mem_location() == MemLocation::UNIFIED_MEM)
      else if (mem_location() == MemLocation::UNIFIED_MEM){
#if ! defined(__CUDA_ARCH__)
        std::cout << "new_memory CPU-GPU ShallowArray_Flat: "
            << get_realname(typeid(T)) << "\n";
        gpuErrorCheck(cudaGetLastError());
#if  defined(USE_PINNED_MEMORY)
        gpuErrorCheck(cudaHostAlloc(&ptr, len, cudaHostAllocDefault));
#else
        gpuErrorCheck(cudaMallocManaged(&ptr, len));
#endif
        gpuErrorCheck(cudaDeviceSynchronize());
#else
        assert(0);
#endif
      }
      else{
        assert(0);
      }
      return ptr;
    }
    CUDA_CALLABLE void delete_memory(T*& ptr)
    {
#if ! defined(__CUDA_ARCH__)
        std::cout << "delete_memory ShallowArray_Flat: "
          << get_realname(typeid(T)) << '\n' << std::flush; 
#endif
      if (_allocated_size == 0)
        return;//do nothing

      //if (Allocator<T, memLocation>::mem_location() == MemLocation::CPU)
      if (mem_location() == MemLocation::CPU)
      {
#if ! defined(__CUDA_ARCH__)
        std::cout << "... delete_memory CPU ShallowArray_Flat: "
          << get_realname(typeid(T)) << '\n'; 
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_STD_ALLOCATOR
        for (size_t i =0; i < length; ++i)
          allocator.destroy(ptr+i);
        allocator.deallocate(ptr, _allocated_size);
#elif defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_PLACEMENT_NEW
        /* we can use multiple thread here as well? */
        for(int i = 0; i < _allocated_size; ++i)
        {
          _data[i].~T();
        }
        delete[] pBuffer;
#else
        {
          ::delete[] ptr;
        }
#endif
#else
        assert(0);
#endif
      }
      //else if (Allocator<T, memLocation>::mem_location() == MemLocation::UNIFIED_MEM)
      else if (mem_location() == MemLocation::UNIFIED_MEM)
      {
#if ! defined(__CUDA_ARCH__)
        std::cout << "... delete_memory CPU-GPU ShallowArray_Flat: " 
            << get_realname(typeid(T)) << "\n";
        gpuErrorCheck(cudaDeviceSynchronize());
#if defined(USE_PINNED_MEMORY)
        gpuErrorCheck(cudaFreeHost(ptr));
#else
        gpuErrorCheck(cudaFree(ptr));         
#endif
#else
        assert(0);
#endif
      }
    }
    //__device__ 
    CUDA_CALLABLE void expand() {
        int new__allocated_size = _allocated_size * 2;
        expand(new__allocated_size);
      }
    CUDA_CALLABLE void expand(int new__allocated_size) {
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_PLACEMENT_NEW
        assert(0);
        T* new_data;
        char* new_pBuffer;
        _data = (T*)new_memory((unsigned long int)new__allocated_size*sizeof(T), &new_pBuffer);  // this use overloaded
        m_end = m_begin;
        _allocated_size = new__allocated_size;
        pBuffer = new_pBuffer;
        return;
#endif
        //_allocated_size *= 2;
        //_allocated_size = new__allocated_size;
        unsigned long int tempLength = (m_end - m_begin);
#ifdef USE_OVERLOADED_NEW
        T* tempBegin = new T[new__allocated_size];
#else
				T* tempBegin = (T*) new_memory(new__allocated_size*sizeof(T));
#endif

#if ! defined(__CUDA_ARCH__)
        //memcpy(tempBegin, m_begin, tempLength * sizeof(T));
        std::copy_n(m_begin, length, tempBegin);
#else
        //cudaMemcpyAsync(void *to, void *from, size, cudaMemcpyDeviceToDevice)
        //cudaMemcpyAsync(tempBegin, m_begin, length*sizeof(T), cudaMemcpyDeviceToDevice);
        assert(0);
#endif
#ifdef USE_OVERLOADED_NEW
        delete[] m_begin;
#else
        delete_memory(m_begin);
#endif
        m_begin = tempBegin;
        m_end = m_begin + tempLength;
        length = static_cast<unsigned long int>(m_end - m_begin);
        _allocated_size = new__allocated_size;
    }

public:
    typedef ShallowArray_Flat<T, memLocation> self_type;
    typedef T value_type;      
    typedef Array_FlatIterator<T, T> iterator;
    typedef Array_FlatIterator<const T, T> const_iterator;
    
    friend class Array_FlatIterator<T, T>;
    friend class Array_FlatIterator<const T, T>;
    /*
    static void *operator new(std::size_t len)
    {
      std::cout << "new in ShallowArray_Flat\n";
    }
    static void *operator new[](std::size_t len)
    {
      std::cout << "new[] in ShallowArray_Flat\n";
    }
    */
    //__device__  
    CUDA_CALLABLE explicit ShallowArray_Flat() : length(0), _allocated_size(0),
    _communicatedSize(0),  _sizeToCommunicate(0), _array_design(FLAT_ARRAY),
      _incremental_size(DEFAULT_INCREMENTAL_SIZE_ARR)
    {
        _mem_location = memLocation;

#if ! defined(__CUDA_ARCH__)
        //if (_mem_location == MemLocation::UNIFIED_MEM and type_is_pointer())
        //  _allocated_size = 0;

        std::cout << "constructor in ShallowArray_Flat: "
          << get_realname(typeid(T)) << '\n'; 
#endif
        m_begin = nullptr;
        m_end = nullptr;
        if (_allocated_size > 0)
        {
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_PLACEMENT_NEW
          {
            char* new_pBuffer;
            /* error as we're trying to bind an address to a reference
             *
             * &new_pBuffer creates a temporary value, which cannot be bound to a reference to non-const.
             * */
            m_begin = (T*)new_memory(_allocated_size * sizeof(T), &new_pBuffer);  // this use overloaded
            pBuffer = new_pBuffer;
            m_end = m_begin;
            return;
          }
#endif
#ifdef USE_OVERLOADED_NEW
          m_begin = new T[_allocated_size];
#else
          m_begin = (T*) new_memory(_allocated_size*sizeof(T));
#endif
          m_end = m_begin;
        }
    }
    CUDA_CALLABLE ShallowArray_Flat(const ShallowArray_Flat* rv)
    {
      //ShallowArray_Flat<T, memLocation>::copyContents(*rv);
      copyContents(*rv);
    }
    //important copy constructor
    CUDA_CALLABLE ShallowArray_Flat(const ShallowArray_Flat& rv)
    {
      //ShallowArray_Flat<T, memLocation>::copyContents(rv);
      copyContents(rv);
    }
    //__device__ 
    //CUDA_CALLABLE const T& operator[] (unsigned int index) const 
    template<typename C>
    CUDA_CALLABLE const T& operator[](C index) const
    {
        return *(m_begin + index);//*(begin+index)
    }
    //__device__ 
    //CUDA_CALLABLE T& operator[] (unsigned int index) 
    template<typename C>
    CUDA_CALLABLE T& operator[](C index)
    {
        return *(m_begin + index);//*(begin+index)
    }
/*
    //__device__ 
    CUDA_CALLABLE T* begin() {
        return m_begin;
    }
    //__device__ 
    CUDA_CALLABLE T* end() {
        return m_end;
    }
*/
    CUDA_CALLABLE iterator begin() 
    {
      return iterator(&m_begin, 0);
    }
    
    CUDA_CALLABLE iterator end() 
    {
      return iterator(&m_begin, size());
    }
    
    CUDA_CALLABLE const_iterator begin() const 
    {
      return const_iterator(const_cast<const T**>(&m_begin), 0);
    }
    
    CUDA_CALLABLE const_iterator end() const 
    {
      return const_iterator(
			    const_cast<const T**>(&m_begin), size());
    }
    //__device__ 
    CUDA_CALLABLE ~ShallowArray_Flat()
    {
#if ! defined(__CUDA_ARCH__)
#ifdef USE_OVERLOADED_NEW
      delete[] m_begin;
#else
      delete_memory(m_begin);
#endif
      m_begin = nullptr;
      m_end = nullptr;
      _allocated_size = 0;
#else
      assert(0);
#endif
    }
    CUDA_CALLABLE ShallowArray_Flat<T, memLocation>& operator=(const ShallowArray_Flat& rv)
    {
      if (this == &rv)
      {
        return *this;
      }
      //Array_Flat<T, memLocation>::operator=(rv);
      destructContents();
      copyContents(rv);
      return *this;
    }
    void duplicate(std::unique_ptr<ShallowArray_Flat<T, memLocation>>& rv) const 
    {
      rv.reset(new ShallowArray_Flat<T, memLocation, blockIncrementSize>(this));
    }
    CUDA_CALLABLE void copyContents(const ShallowArray_Flat& rv)
    {
        if (length == 0 and rv.length == 0)
          return;
        //internalCopy(_data, rv._data); // multiplicating is done if necessary
        ////TUAN TODO: fix this as this is slow
        if (rv.length > 0)
        {
          if (length > 0)
            destructContents();
          //_mem_location = rv._mem_location;
          _array_design = rv._array_design;
          //_data = new T[sizeof(T) * length];
          //_data = new((size_t)sizeof(T) * length);
          //_data = (T*)new_memory((size_t)sizeof(T) * length);
          //_allocated_size = length;
          resize_allocated(rv.length, 0);
          length = rv.length;

          //_incremental_size = rv._incremental_size; 
          //_communicatedSize = rv._communicatedSize;  
          //_sizeToCommunicate = rv._sizeToCommunicate;
          //for (size_t j = 0; (j < length); j++) {
          //  /* copy individual elements*/
          //  internalCopy(m_begin[j], rv.m_begin[j]); // multiplicating is done if necessary
          //};
          //unsigned long int tempLength = (rv.m_end - rv.m_begin);
#if ! defined(__CUDA_ARCH__)
          //memcpy(tempBegin, m_begin, tempLength * sizeof(T));
          std::copy_n(rv.m_begin, rv.length, &m_begin[0]);
#else
          assert(0);
          //cudaMemcpyAsync(m_begin, rv.m_begin, length*sizeof(T), cudaMemcpyDeviceToDevice);
#endif
          m_end = m_begin+length;
        }
    }
    CUDA_CALLABLE void destructContents()
    {
      if (m_begin != nullptr)
      {
        //delete m_begin;
        delete_memory(m_begin);
        m_begin = nullptr;
        m_end = nullptr;
      }
      length = 0;
      _allocated_size = 0;
    }
    void internalCopy(T& lval, T& rval)
    {
      lval = rval;
    }

    CUDA_CALLABLE T* getDataRef() {return m_begin; };
    CUDA_CALLABLE void push_back(const T& element) 
    {
      insert(element);
    }
    CUDA_CALLABLE void insert(const T& element){
      //add(element);
      if (length >= _allocated_size)
      {
        //resize_allocated(_allocated_size + _incremental_size); 
        expand(_allocated_size + _incremental_size);
      }
#ifdef USE_OVERLOADED_NEW
      new (m_end) T(element);
      m_end++;
      length++;
#else
      m_begin[length++] = element;
      m_end++;
#endif
    }
    void assign(int64_t n, const T& val)
    {
      //to be implemented
      assert(0);
    }
    void replace(int64_t index, const T& element) //replace the element at index 'index' with new value
    {
      //to be implemented
      assert(0);
    }
    //__device__ 
    CUDA_CALLABLE void add(T t) {
      insert(t);
    }
    //__device__ 
    CUDA_CALLABLE T pop() {
      T endElement = (*m_end);
#ifdef USE_OVERLOADED_NEW
      delete m_end;
#else
      assert(0);
#endif
      m_end--;
      return endElement;
    }
    CUDA_CALLABLE void increase()
    {
      if ((m_end - m_begin) >= _allocated_size) {
        expand();
      }
      /* TODO  revise this */
#ifdef USE_OVERLOADED_NEW
      new (m_end) T();
      m_end++;
      length++;
#else
      m_end++;
      length++;
#endif
    }
    void decrease()
    {
      //to be implemented
      assert(0);
    }
    void demote (unsigned long int, unsigned long int)
    {
      //to be implemented
      assert(0);
    }

    CUDA_CALLABLE inline unsigned long int allocated_size() const 
    {
      return _allocated_size;
    }
    CUDA_CALLABLE inline unsigned long int size() const 
    {
      return length;
    }
    __device__ unsigned long int getSize() {
        return length;
    }
    void increaseSizeTo(int64_t newSize, bool force_trim_memory_to_smaller = false)
    {
      resize_allocated(newSize, force_trim_memory_to_smaller);
      assert(newSize < _allocated_size);
      length = newSize;
    }
    void decreaseSizeTo(int64_t newSize)
    {
      //to be implemented
      assert(0);
    }
    void resize_allocated_subarray(int64_t MAX_SUBARRAY_SIZE, uint8_t mem_location)
    {
      auto newSize = MAX_SUBARRAY_SIZE;
      resize_allocated(newSize, 0);
    }
    void resize_allocated(int64_t newSize, bool force_trim_memory_to_smaller = false)
		{
      if (newSize <= (int64_t)_allocated_size-NUM_TRAILING_ELEMENT)
      {
        if (! force_trim_memory_to_smaller)
          return;
      }
      if (newSize < 0)
      {
        assert(0);
        return;
      }
      unsigned long int num_data_2_allocate = newSize+NUM_TRAILING_ELEMENT;
      //_allocated_size = num_data_2_allocate;
      //expand(_allocated_size);
      expand(num_data_2_allocate);
    }
    template <typename C> static bool _inner_type_is_pointer(int)
    {
      return std::is_pointer<C>::value;
    }
    static bool type_is_pointer()
    { 
      return _inner_type_is_pointer<T>(0); 
    }
    void sort()
    {
      //to be implemented
      assert(0);
    };
    void unique()
    {
      //to be implemented
      assert(0);
    };
    void merge(const ShallowArray_Flat& rv)
    {
      //to be implemented
      assert(0);
    };
    void clear()
    {
      //to be implemented
      assert(0);
    };
    const unsigned & getCommunicatedSize() const
    {
      return _communicatedSize;
    }

    unsigned & getCommunicatedSize()
    {
      return _communicatedSize;
    }

    void setCommunicatedSize(unsigned communicatedSize)
    {
      _communicatedSize = communicatedSize;
    }

    const unsigned & getSizeToCommunicate() const
    {
      return _sizeToCommunicate;
    }

    unsigned & getSizeToCommunicate()
    {
      return _sizeToCommunicate;
    }

    void setSizeToCommunicate(unsigned sizeToCommunicate)
    {
      _sizeToCommunicate = sizeToCommunicate;
    }
};
template <typename T, int memLocation>
std::ostream& operator<<(std::ostream& os, ShallowArray_Flat<T, memLocation>& arr) {
   auto size = arr.size();
   for (auto i = 0; i < size; ++i) {
      os << arr[i] << " ";
   }
   return os;
}

template <typename T, int memLocation>
std::istream& operator>>(std::istream& is, ShallowArray_Flat<T, memLocation>& arr) {
   assert(0);
   return is;
}

//template <typename T, int memLocation, unsigned blockIncrementSize = SUGGESTEDBLOCKINCREMENTSIZE>
//class ShallowArray_Flat : public Managed
//{
//  public:
//    typedef ShallowArray_Flat<T, memLocation> self_type;
//    typedef T value_type;      
//    typedef Array_FlatIterator<T, T> iterator;
//    typedef Array_FlatIterator<const T, T> const_iterator;
//    
//    friend class Array_FlatIterator<T, T>;
//    friend class Array_FlatIterator<const T, T>;
//
//  CUDA_CALLABLE ShallowArray_Flat();
//  CUDA_CALLABLE ShallowArray_Flat(const ShallowArray_Flat* rv);
//  CUDA_CALLABLE ShallowArray_Flat(const ShallowArray_Flat& rv); //important copy constructor
//  CUDA_CALLABLE ShallowArray_Flat& operator=(const ShallowArray_Flat& rv);
//#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_PLACEMENT_NEW
//    T* new_memory(unsigned long int len, char** new_pBuffer)
//#else
//    T * new_memory(unsigned long int len)
//#endif
//    {
//      T* ptr=0; //void *ptr;
//      if (_mem_location == MemLocation::CPU)
//      {
////#define DEBUG_SIZE_INFO
//#if defined(DEBUG_SIZE_INFO)
//	double sizeGB = ((double)len) /1024/1024/1024;
//	double sizeMB = ((double)len) /1024/1024;
//	if (sizeGB >= 1)
//	{
//	  std::cout << "AAA" << sizeGB << "GB for " 
//	    //<< typeid(*this).name() << " " 
//	    << typeid(T).name() << "\n";
//	}
//	else if (sizeMB >= 1)
//	{
//	  std::cout << "BBB" << sizeMB << "MB for " 
//	    << typeid(T).name() << "\n";
//	}
//#endif
//#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_STD_ALLOCATOR
//	{
//	  ptr = allocator.allocate(len/sizeof(T));
//	}
//#elif defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_PLACEMENT_NEW
//	{
//	/*
//	 https://stackoverflow.com/questions/15254/can-placement-new-for-arrays-be-used-in-a-portable-way
//	 */
//	size_t NUMELEMENTS = len/sizeof(T);
//	//char* pBuffer;
//	//pBuffer = malloc(len);
//	//pBuffer = new char[len];
//	*new_pBuffer = new char[len];
//	/* placement new on array */
//	//ptr = new (pBuffer) T[len/sizeof(T)];
//	/* placement new on individual element */
//	/* we can use multiple thread here as well? */
//	//ptr = (T*)pBuffer;
//#if defined(DEBUG_SIZE_INFO)
//	if (sizeGB >= 1)
//	{
//	  std::cout << "    pass new char" << "\n";
//	}
//#endif
//	//
//	ptr = (T*)*new_pBuffer;
//	/* serisal version */
//	//https://stackoverflow.com/questions/4011577/placement-new-array-alignment?rq=1
//	//https://stackoverflow.com/questions/4754763/object-array-initialization-without-default-constructor
//	
//	for(size_t i = 0; i < NUMELEMENTS; ++i)
//	{
//	  //&ptr[i] = new (ptr + i) T();
//	  new (ptr + i) T();
//        #if defined(DEBUG_SIZE_INFO)
//	  if (sizeGB >= 1)
//	  {
//	    if (i > 0 and i % 100000000 == 0)
//	      std::cout << "    index "<< i << "\n";
//	  }
//        #endif
//       }
//       //end serial version
//       
//	/* parallel version */
////       const size_t nthreads = std::min(10, (int)std::thread::hardware_concurrency());
////       {
////	  // Pre loop
////	  //std::cout<<"parallel ("<<nthreads<<" threads):"<<std::endl;
////	  std::vector<std::thread> threads(nthreads);
////	  std::mutex critical;
////	  const size_t nloop = NUMELEMENTS; 
////	  for(int t = 0;t<nthreads;t++)
////	  {
////	     /* each thread bind to a lambdas function
////		   the lambdas function accepts 3 arguments: 
////		      t= thread index
////		      bi = start index of data
////		      ei = end index of data
////		*/
////	     threads[t] = std::thread(std::bind(
////	     [&](const size_t bi, const size_t ei, const int t)
////	     {
////		// loop over all items
////		for(size_t gn = bi; gn <ei; gn++)
////		{
////		  new (ptr + gn) T();
////		}
////		{
////		//critical region
////		std::lock_guard<std::mutex> lock(critical);
////
////#if defined(DEBUG_SIZE_INFO)
////	if (sizeGB >= 1)
////	{
////	    std::cout << "    done thread "<< t << "\n";
////	}
////#endif
////		}
////	     }, t*nloop/nthreads, (t+1)==nthreads?nloop:(t+1)*nloop/nthreads, t));
////	  }
////	  std::for_each(threads.begin(),threads.end(),[](std::thread& x){x.join();});
////	  // Post loop
////	  // ..nothing
//////#ifdef DEBUG_TIMER 
//////       sim->benchmark_timelapsed_diff("... time for multithread() 20threads" );
//////#endif
////       }
//	}
//#else
//	{
//	//ptr = ::new T[len/sizeof(T)];
//	//ptr = ::new T[len/sizeof(T)];
//	ptr = new T[len/sizeof(T)];
//	//size_t numObjs = len/sizeof(T);
//	//ptr = new T[numObjs];
//	}
//#endif
//#if defined(DEBUG_SIZE_INFO)
//	if (sizeGB >= 1)
//	{
//	  std::cout << "end" << sizeGB << "GB\n";
//	}
//	else if (sizeMB >= 1)
//	{
//	  std::cout << "end" << sizeMB << "MB\n";
//	}
//#endif
//      }
//      else if (_mem_location == MemLocation::UNIFIED_MEM){
//	gpuErrorCheck(cudaGetLastError());
//#if  defined(USE_PINNED_MEMORY)
//	gpuErrorCheck(cudaHostAlloc(&ptr, len, cudaHostAllocDefault));
//#else
//	gpuErrorCheck(cudaMallocManaged(&ptr, len));
//#endif
//	gpuErrorCheck(cudaDeviceSynchronize());
//      }
//      else{
//	assert(0);
//      }
//      return ptr;
//    }
//    void delete_memory(T*& ptr)
//    {
//      if (_mem_location == MemLocation::CPU)
//      {
//#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_STD_ALLOCATOR
//	for (size_t i =0; i < _size; ++i)
//	  allocator.destroy(ptr+i);
//	allocator.deallocate(ptr, _allocated_size);
//#elif defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_PLACEMENT_NEW
//	/* we can use multiple thread here as well? */
//	for(int i = 0; i < _allocated_size; ++i)
//	{
//	  _data[i].~T();
//	}
//	delete[] pBuffer;
//#else
//	{
//	  ::delete[] ptr;
//	}
//#endif
//      }
//      else if (_mem_location == MemLocation::UNIFIED_MEM){
//#if ! defined(__CUDA_ARCH__)
//	gpuErrorCheck(cudaDeviceSynchronize());
//#if defined(USE_PINNED_MEMORY)
//	gpuErrorCheck(cudaFreeHost(ptr));
//#else
//	gpuErrorCheck(cudaFree(ptr));         
//#endif
//#endif
//      }
//    }
//  //virtual void duplicate(
//  //    std::unique_ptr<Array_Flat<T> >& rv) const;
//  virtual void duplicate(
//      std::unique_ptr<Array_Flat<T, memLocation> >& rv) const;
//  //virtual void duplicate(
//  //    std::unique_ptr<ShallowArray_Flat<T, memLocation> >& rv) const;
//  virtual void duplicate(
//      std::unique_ptr<ShallowArray_Flat<T, memLocation, blockIncrementSize> >& rv) const;
//  CUDA_CALLABLE virtual ~ShallowArray_Flat();
//
//  protected:
//  virtual void internalCopy(T& lval, T& rval);
//  void destructContents();
//  void copyContents(const ShallowArray_Flat& rv);
//};

#else
template <typename T, int memLocation=0, unsigned blockIncrementSize = SUGGESTEDBLOCKINCREMENTSIZE>
class ShallowArray_Flat : public Array_Flat<T, memLocation>
{
  public:
  CUDA_CALLABLE ShallowArray_Flat();
  CUDA_CALLABLE ShallowArray_Flat(const ShallowArray_Flat* rv);
  CUDA_CALLABLE ShallowArray_Flat(const ShallowArray_Flat& rv); //important copy constructor
  CUDA_CALLABLE ShallowArray_Flat& operator=(const ShallowArray_Flat& rv);
  //virtual void duplicate(
  //    std::unique_ptr<Array_Flat<T> >& rv) const;
  virtual void duplicate(
      std::unique_ptr<Array_Flat<T, memLocation> >& rv) const;
  //virtual void duplicate(
  //    std::unique_ptr<ShallowArray_Flat<T, memLocation> >& rv) const;
  virtual void duplicate(
      std::unique_ptr<ShallowArray_Flat<T, memLocation, blockIncrementSize> >& rv) const;
  CUDA_CALLABLE virtual ~ShallowArray_Flat();
  CUDA_CALLABLE void destructContents();

  protected:
  CUDA_CALLABLE virtual void internalCopy(T& lval, T& rval);
  CUDA_CALLABLE void copyContents(const ShallowArray_Flat& rv);
};

template <typename T, int memLocation, unsigned blockIncrementSize>
CUDA_CALLABLE ShallowArray_Flat<T, memLocation, blockIncrementSize>::ShallowArray_Flat()
    : Array_Flat<T, memLocation>(blockIncrementSize)
{
}

template <typename T, int memLocation, unsigned blockIncrementSize>
CUDA_CALLABLE
ShallowArray_Flat<T, memLocation, blockIncrementSize>::ShallowArray_Flat(
    const ShallowArray_Flat* rv)
//   : Array_Flat<T, memLocation>(rv) // can not do this because of the pure virtual method in
//   copyContents
{
  Array_Flat<T, memLocation>::copyContents(*rv);
  copyContents(*rv);
}

template <typename T, int memLocation, unsigned blockIncrementSize>
CUDA_CALLABLE
ShallowArray_Flat<T, memLocation, blockIncrementSize>::ShallowArray_Flat(
    const ShallowArray_Flat& rv)
//   : Array_Flat<T, memLocation>(rv) // can not do this because of the pure virtual method in
//   copyContents
{
  Array_Flat<T, memLocation>::copyContents(rv);
  copyContents(rv);
}

template <typename T, int memLocation, unsigned blockIncrementSize>
CUDA_CALLABLE
ShallowArray_Flat<T, memLocation, blockIncrementSize>&
    ShallowArray_Flat<T, memLocation, blockIncrementSize>::
        operator=(const ShallowArray_Flat& rv)
{
  if (this == &rv)
  {
    return *this;
  }
  destructContents();
  Array_Flat<T, memLocation>::operator=(rv);
  copyContents(rv);
  return *this;
}

//template <typename T, int memLocation, unsigned blockIncrementSize>
//void ShallowArray_Flat<T, memLocation, blockIncrementSize>::duplicate(
//    std::unique_ptr<Array_Flat<T> >& rv) const
//{
//  rv.reset(new ShallowArray_Flat<T, memLocation, blockIncrementSize>(this));
//}

template <typename T, int memLocation, unsigned blockIncrementSize>
void ShallowArray_Flat<T, memLocation, blockIncrementSize>::duplicate(
    std::unique_ptr<Array_Flat<T, memLocation> >& rv) const
{
  rv.reset(new ShallowArray_Flat<T, memLocation, blockIncrementSize>(this));
}

//template <typename T, int memLocation, unsigned blockIncrementSize>
//void ShallowArray_Flat<T, memLocation, blockIncrementSize>::duplicate(
//    std::unique_ptr<ShallowArray_Flat<T, memLocation> >& rv) const
//{
//  rv.reset(new ShallowArray_Flat<T, memLocation, blockIncrementSize>(this));
//}

template <typename T, int memLocation, unsigned blockIncrementSize>
void ShallowArray_Flat<T, memLocation, blockIncrementSize>::duplicate(
    std::unique_ptr<ShallowArray_Flat<T, memLocation, blockIncrementSize> >& rv) const
{
  rv.reset(new ShallowArray_Flat<T, memLocation, blockIncrementSize>(this));
}

template <typename T, int memLocation, unsigned blockIncrementSize>
CUDA_CALLABLE
ShallowArray_Flat<T, memLocation, blockIncrementSize>::~ShallowArray_Flat()
{
  destructContents();
}

template <typename T, int memLocation, unsigned blockIncrementSize>
CUDA_CALLABLE
void ShallowArray_Flat<T, memLocation, blockIncrementSize>::internalCopy(T& lval,
                                                                  T& rval)
{
  lval = rval;
}

template <typename T, int memLocation, unsigned blockIncrementSize>
CUDA_CALLABLE
void ShallowArray_Flat<T, memLocation, blockIncrementSize>::copyContents(
    const ShallowArray_Flat& rv)
{
}

template <typename T, int memLocation, unsigned blockIncrementSize>
CUDA_CALLABLE
void ShallowArray_Flat<T, memLocation, blockIncrementSize>::destructContents()
{
  Array_Flat<T, memLocation>::destructContents();
}

#endif
#endif
