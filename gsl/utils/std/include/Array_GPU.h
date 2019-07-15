#ifndef ARRAY_GPU_H
#define ARRAY_GPU_H
#include <type_traits>
#include <algorithm>
#include <cstddef>
#include "rndm.h"
#include <vector>
//#define USE_SMART_PTR
#ifdef USE_SMART_PTR
#include "Array_GPU_unique_ptr.h"
#else
//class GranuleMapper;

#define MINIMAL_SIZE_ARR 12
// use BLOCK_INCREMENTAL  (which can be 4, 8 or 16)
#define DEFAULT_INCREMENTAL_SIZE_ARR 4
/* element reserved for testing iterator vs .end()*/
#define NUM_TRAILING_ELEMENT 1

// use flat-array on Unified Memory
// (if not defined, use flat-array on CPU memory)
//#define USE_FLATARRAY_UM



template<template<class> class T, class U>
struct isDerivedFrom
{
private:
    template<class V>
    static decltype(static_cast<const T<V>&>(std::declval<U>()), std::true_type{})
    test(const T<V>&);

    static std::false_type test(...);
public:
    static constexpr bool value = decltype(isDerivedFrom::test(std::declval<U>()))::value;
};

//template<template<class> class T, template<class> class U>
//struct isDerivedFromTemplate
//{
//private:
//    template<class V>
//    static decltype(static_cast<const T<V>&>(std::declval<U>()), std::true_type{})
//    test(const T<V>&);
//
//    static std::false_type test(...);
//public:
//    static constexpr bool value = decltype(isDerivedFrom::test(std::declval<U>()))::value;
//};

class Managed
{
  public:
    /* 
     * T = data type for 1 element
     * len = num_elements * sizeof(T)
     */
    //void *operator new(size_t len)
    //{
    //  void *ptr;
    //  cudaMallocManaged(&ptr, len);
    //  cudaDeviceSynchronize();
    //  return ptr;
    //}
    //void operator delete(void* ptr)
    //{
    //  cudaDeviceSynchronize();
    //  cudaFree(ptr);
    //}

    void * new_memory(size_t len)
    {
      void *ptr;
      gpuErrorCheck(cudaDeviceSynchronize());
      gpuErrorCheck(cudaMallocManaged(&ptr, len));
      return ptr;
    }
    void delete_memory(void* ptr)
    {
      gpuErrorCheck(cudaDeviceSynchronize());
      gpuErrorCheck(cudaFree(ptr));
    }
};

#include <memory>
#include <new>
/*
 * CONSTRAINT: 
 *   to enable the iterrator to work properly, _data is at minimal 1 element
 */
template <class T, int memLocation = 0>
class Array_Flat //: public Managed
{
    //void _check(cudaError_t r, int line) {
    //  if (r != cudaSuccess) {
    //    printf("CUDA error on line %d: %s\n", line, cudaGetErrorString(r));
    //    exit(0);
    //  }
    //}
 public:
    typedef Array_Flat<T, memLocation> self_type;
    typedef T value_type;      
    typedef Array_FlatIterator<T, T> iterator;
    typedef Array_FlatIterator<const T, T> const_iterator;
    
    friend class Array_FlatIterator<T, T>;
    friend class Array_FlatIterator<const T, T>;
    /* len = in number of bytes */
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_PLACEMENT_NEW
    //T* new_memory(size_t len, char*& new_pBuffer)
    T* new_memory(size_t len, char** new_pBuffer)
#else
    T * new_memory(size_t len)
#endif
    {
      //std::cout << " ... new_memory size in GB " << len/1024/1024/1024 << " GB\n";
      /* testing: use again host allocation */
      T* ptr; //void *ptr;
//#ifdef USE_FLATARRAY_UM
//      cudaMallocManaged(&ptr, len);
//      cudaDeviceSynchronize();
//#else
//      ptr = ::new T[len/sizeof(T)];
//#endif

      if (_mem_location == MemLocation::CPU)
      {
//#define DEBUG_SIZE_INFO
#if defined(DEBUG_SIZE_INFO)
	double sizeGB = ((double)len) /1024/1024/1024;
	double sizeMB = ((double)len) /1024/1024;
	if (sizeGB >= 1)
	{
	  std::cout << "AAA" << sizeGB << "GB for " 
	    //<< typeid(*this).name() << " " 
	    << typeid(T).name() << "\n";
	}
	else if (sizeMB >= 1)
	{
	  std::cout << "BBB" << sizeMB << "MB for " 
	    << typeid(T).name() << "\n";
	}
#endif
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_STD_ALLOCATOR
	ptr = allocator.allocate(len/sizeof(T));
#elif defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_PLACEMENT_NEW
	/*
	 https://stackoverflow.com/questions/15254/can-placement-new-for-arrays-be-used-in-a-portable-way
	 */
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
#else
	//ptr = ::new T[len/sizeof(T)];
	//ptr = ::new T[len/sizeof(T)];
	ptr = new T[len/sizeof(T)];
	//size_t numObjs = len/sizeof(T);
	//ptr = new T[numObjs];
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
      }
      else if (_mem_location == MemLocation::UNIFIED_MEM){
      gpuErrorCheck(cudaGetLastError());
      gpuErrorCheck(cudaMallocManaged(&ptr, len));
      gpuErrorCheck(cudaDeviceSynchronize());
      }
      else{
	assert(0);
      }
      return ptr;
    }
    void delete_memory(T*& ptr)
    {
//#ifdef USE_FLATARRAY_UM
//      cudaDeviceSynchronize();
//      cudaFree(ptr);
//#else
//      ::delete[] ptr;
//#endif
      if (_mem_location == MemLocation::CPU)
      {
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_STD_ALLOCATOR
	for (size_t i =0; i < _size; ++i)
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
      ::delete[] ptr;
#endif
      }else if (_mem_location == MemLocation::UNIFIED_MEM){
      gpuErrorCheck(cudaDeviceSynchronize());
      gpuErrorCheck(cudaFree(ptr));         
      }
    }

    //bool should_be_on_UM() {return true;};
    /* TUAN TODO probably safe to remove?
     * check if something likfe ShallowArray_Flat(a_number) or 
     * DeepPointerArray_Flat(a_number) is being used anywhere
     * YES: 
     *      DuplicatePointerArray<GranuleMapper, 50> _granuleMapperList;
     * With MemLocation, now we use
     *      DuplicatePointerArray<GranuleMapper, 0, 50> _granuleMapperList;
     */
    Array_Flat(int64_t incrementSize);
    /* delete all existing memory
     * then re-create to the minimal size
     * and set num-elements to 0
     */
    virtual void clear() {
      if (_allocated_size > NUM_TRAILING_ELEMENT)
      {
	//resize_allocated(NUM_TRAILING_ELEMENT, true);
	resize_allocated(0, true);
      }
      _size = 0;
    };

    CUDA_CALLABLE T* getDataRef() {return _data; };
    void increaseSizeTo(int64_t newSize, bool force_trim_memory_to_smaller = false);
    void decreaseSizeTo(int64_t newSize);
    //void resize_allocated(unsigned newSize);
    void resize_allocated(int64_t newSize, bool force_trim_memory_to_smaller = false);
    void resize_allocated_subarray(int64_t MAX_SUBARRAY_SIZE, uint8_t location);
    void increase();
    void decrease();
    void assign(int64_t n, const T& val);
    void insert(const T& element);
    void replace(int64_t index, const T& element); //replace the element at index 'index' with new value
    void push_back(const T& element) 
    {
      insert(element);
    }
    CUDA_CALLABLE T& operator[](int64_t index);
    CUDA_CALLABLE const T& operator[](int64_t index) const;
    CUDA_CALLABLE Array_Flat& operator=(const Array_Flat& rv);
    //virtual void duplicate(std::unique_ptr<Array_Flat<T>>& rv) const = 0;
    virtual void duplicate(std::unique_ptr<Array_Flat<T, memLocation>>& rv) const = 0;
    virtual ~Array_Flat();
    
    CUDA_CALLABLE size_t size() const 
    {
      return _size;
    }
    CUDA_CALLABLE size_t allocated_size() const 
    {
      return _allocated_size;
    }
    
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

    CUDA_CALLABLE iterator begin() 
    {
      return iterator(&_data, 0);
    }
    
    CUDA_CALLABLE iterator end() 
    {
      return iterator(&_data, size());
    }
    
    CUDA_CALLABLE const_iterator begin() const 
    {
      return const_iterator(const_cast<const T**>(&_data), 0);
    }
    
    CUDA_CALLABLE const_iterator end() const 
    {
      return const_iterator(
			    const_cast<const T**>(&_data), size());
    }
     
    void sort();
    void unique();
    void merge(const Array_Flat& rv);

   protected:

      // This is required for the copy constructor of derived classes, 
      // the copy constructor of this class is really copyContents which uses
      // a virtual method that is implemented by the inheriting classes,
      // copyContents is also called by the inheriting classes
      Array_Flat()    
	: _allocated_size(0), _incremental_size(DEFAULT_INCREMENTAL_SIZE_ARR), 
	_size(0), _communicatedSize(0),  _sizeToCommunicate(0),
	//_mem_location(MemLocation::CPU), _array_design(FLAT_ARRAY)
	_mem_location(memLocation), _array_design(FLAT_ARRAY)
      {
	_data = nullptr;
	//TUAN TODO : plan to disable this, and any use must call 'resize_allocated()' explitly
	//STATUS: completed
	//So, for now we only pre-allocate if it's on CPU
	if (_mem_location == MemLocation::CPU)
	  resize_allocated(MINIMAL_SIZE_ARR);
      }

      virtual void internalCopy(T& lval, T& rval) = 0;
      /* wipe everything
       * don't create minimal data
       */
      CUDA_CALLABLE void destructContents();
      CUDA_CALLABLE void copyContents(const Array_Flat& rv);
      void demote (size_t, size_t); 
	  // NOTE: Arrays are organized in the form of multiple 'logical blocks'
	  //       i.e. memory increase/reduced, in the form of one or many blocks
      size_t _allocated_size;  // the #elements as allocated
      int _incremental_size;  // the extra #elements to be allocated
      size_t _size; //the number of elements in the array containing data
      unsigned _communicatedSize; //the number of elements MPI has communicated
      unsigned _sizeToCommunicate; //the number of elements MPI is to communicate
      T* _data; // the flat array of data (to be on Unified Memory)
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_PLACEMENT_NEW
      char* pBuffer;
#endif
      uint8_t _mem_location;  //
      uint8_t _array_design;
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_STD_ALLOCATOR
      std::allocator<T> allocator;
#endif
   public:
      //TUAN: make this public so that we can use
      // Array_Flat<int>::MemLocation::CPU
      typedef enum {CPU=0, UNIFIED_MEM} MemLocation; //IMPORTANT: Keep CPU=0
      typedef enum {DOUBLE_ARRAY, FLAT_ARRAY} ArrayDesign;
};

template <class T, int memLocation>
Array_Flat<T, memLocation>::Array_Flat(int64_t incrementSize)
  : _allocated_size(0), _incremental_size(incrementSize), 
  _size(0), _communicatedSize(0),  _sizeToCommunicate(0),
  _mem_location(memLocation), _array_design(FLAT_ARRAY)
{
   if (_incremental_size == 0)
   {
     _incremental_size = DEFAULT_INCREMENTAL_SIZE_ARR; //default
   }
   _data = nullptr;
   //TUAN TODO : plan to disable this, and any use must call 'resize_allocated()' explitly
   //STATUS: completed
   //So, for now we only pre-allocate if it's on CPU
   if (_mem_location == MemLocation::CPU)
     resize_allocated(MINIMAL_SIZE_ARR);
   //_data = new T[sizeof(T) * num_elements];
   ////if (std::is_same<T, GranuleMapper>::value) { 
   ////  /* ... */ 
   ////_data = new T[sizeof(T) * num_elements];
   ////}
   ////else{
   ////  _data = static_cast<T*>(::operator new(sizeof(T) * num_elements));
   ////}
}

/*
 * IMPORTANT:
 * newSize = reflect the number of elements it can accommodate;
 *    but the underlying memory allocated can be larger to support proper memory handling of iterator
 * just to make sure the allocated memory _allocated_size >=  'newSize'
 * IMPORTANT: do not change '_size'
 * if larger, maintain the existing data and physically allocate more if needed
 * if smaller, 
 *     (default)just trim the elements at the end [no change physically allocated memory]
 */
template <class T, int memLocation>
void Array_Flat<T, memLocation>::resize_allocated(int64_t newSize, bool force_trim_memory_to_smaller)
{
  //std::cout << " resize_allocated size in GB " << newSize * sizeof(T)/1024/1024/1024 << " GB\n";
  /* As 
   * ShallowArray_Flat< type_element>
   *   if type_element is 'ShallowArray'
   *   then cudaMalloc or cudaMallocManaged
   *   does not initialize data members 
   *  SO: we need to check this
   */
  if (_incremental_size == 0)
  {
    /* 
     * It means Array_Flat is the type of an element in the Array_Flat, i.e. 
     * 		ShallowArray_Flat<ShallowArray_Flat<int>* > 
     */
    //_incremental_size = DEFAULT_INCREMENTAL_SIZE_ARR; //default
    //if (std::is_same<T,  Array_Flat>::value)
    //{
    //  /* need to allocate memory */
    //  for (int i = 0; i < num_data_2_allocate; i++){
    //    //new_data[i] = new_memory((size_t))
    //  }
    //}
  }
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
  size_t num_data_2_allocate = newSize+NUM_TRAILING_ELEMENT;
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_PLACEMENT_NEW
  T* new_data;
  char* new_pBuffer;
  /* error as we're trying to bind an address to a reference 
   *
   * &new_pBuffer creates a temporary value, which cannot be bound to a reference to non-const.
   * */
  //new_data = (T*)new_memory(num_data_2_allocate*sizeof(T), &new_pBuffer);  // this use overloaded
  new_data = (T*)new_memory(num_data_2_allocate*sizeof(T), &new_pBuffer);  // this use overloaded
  //char** pnew_pBuffer = &new_pBuffer;
  //new_data = (T*)new_memory(num_data_2_allocate*sizeof(T), pnew_pBuffer);  // this use overloaded
#else
  //T* new_data = new T[newSize*sizeof(T)];  // this use standard C++ 
  //T* new_data = new((size_t)newSize*sizeof(T));  // this use overloaded
  T* new_data = (T*)new_memory(num_data_2_allocate*sizeof(T));  // this use overloaded
#endif
//  /* TODO
//   * check if 'T' is of type Array_Flat or ShallowArray_Flat
//   * then allocate new_data[i]
//   */
//  //if (isDerivedFrom<Array_Flat,  decltype(new_data[0])>::value)
//  if (isDerivedFrom<Array_Flat,  T>::value)
//  {
//#define MAX_SUBARRAY_SIZE 20
//    /* need to allocate memory */
//    for (int i = 0; i < num_data_2_allocate; i++){
//      new_data[i].resize_allocated(MAX_SUBARRAY_SIZE);
//    }
//  }

  /* IDEA
   * Maybe we overwirite std::allocator_traits
   * so that we use the information of so-called
   *   MAX_SUBARRAY_SIZE
   * and allocate 
   *    num_data_2_allocate * MAX_SUBARRAY_SIZE
   * then call new_memory(on that memory region) for each 
   *    new_data[i]  which is a sub-array
   */

  //FINALLY
  if (_data != nullptr)
  {
    if (_size > 0)
      std::copy_n((T*)_data, std::min((size_t)newSize, _size), new_data);
    delete_memory(_data);  // this use overloadded 
  }
  _data = new_data;
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_PLACEMENT_NEW
  pBuffer = new_pBuffer;
#endif
  _allocated_size = num_data_2_allocate;
}

template <class T, int memLocation>
void Array_Flat<T, memLocation>::resize_allocated_subarray(int64_t MAX_SUBARRAY_SIZE,
   uint8_t mem_location)
{
  if (MAX_SUBARRAY_SIZE> std::numeric_limits<decltype(_incremental_size)>::max())
  {
    std::cout << "ERROR: size too large\n";
    assert(0);
    return;
  }
//#define MAX_SUBARRAY_SIZE 20
  /* need to allocate memory */
  auto newSize = MAX_SUBARRAY_SIZE;
  _mem_location = mem_location;
  if (_incremental_size == 0)
  {
    _incremental_size = newSize;
    _mem_location = mem_location;
    /* 
     * It means Array_Flat is the type of an element in the Array_Flat, i.e. 
     * 		ShallowArray_Flat<ShallowArray_Flat<int>* > 
     */
    //_incremental_size = DEFAULT_INCREMENTAL_SIZE_ARR; //default
    //if (std::is_same<T,  Array_Flat>::value)
    //{
    //  /* need to allocate memory */
    //  for (int i = 0; i < num_data_2_allocate; i++){
    //    //new_data[i] = new_memory((size_t))
    //  }
    //}
  }
  resize_allocated(newSize, 1);
}

/*
 * make an array (i.e.i adjust _size) larger or smaller
 */
template <class T, int memLocation>
void Array_Flat<T, memLocation>::increaseSizeTo(int64_t newSize, bool force_trim_memory_to_smaller)
{
  //resize_allocated(newSize+NUM_TRAILING_ELEMENT, force_trim_memory_to_smaller);
  resize_allocated(newSize, force_trim_memory_to_smaller);
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_STD_ALLOCATOR
  for (size_t i = _size; i < newSize; ++i)
    allocator.construct(_data + i);
#endif
  _size = newSize;
}


template <class T, int memLocation>
void Array_Flat<T, memLocation>::decreaseSizeTo(int64_t newSize)
{
  //resize_allocated(newSize+NUM_TRAILING_ELEMENT);
  resize_allocated(newSize);
  _size = newSize;
}

/*
 * increase the logical value _size 1 element
 * and if needed, make more space
 */
template <class T, int memLocation>
void Array_Flat<T, memLocation>::increase()
{
  resize_allocated(_size+1);
  _size += 1;
}

template <class T, int memLocation>
void Array_Flat<T, memLocation>::decrease()
{
  resize_allocated(_size-1);
  _size -= 1;
}

template <class T, int memLocation>
void Array_Flat<T, memLocation>::insert(const T& element)
{
  if (_allocated_size == 0)
  {
#if defined(FLAT_MEM_MANAGEMENT) && FLAT_MEM_MANAGEMENT  == USE_PLACEMENT_NEW
  T* new_data;
  char* new_pBuffer;
  _data = (T*)new_memory((size_t)_incremental_size*sizeof(T), &new_pBuffer);  // this use overloaded
#else
    //_data = new T[_incremental_size*sizeof(T)];
    //_data = new((size_t)_incremental_size*sizeof(T));  // this use overloaded
    _data = (T*)new_memory((size_t)_incremental_size*sizeof(T));  // this use overloaded
#endif
    _allocated_size = _incremental_size;
  }
  if (_size <= _allocated_size)
    _data[_size++] = element;
  else{
    resize_allocated(_allocated_size + _incremental_size); 
    _data[_size++] = element;
  }
}

template <class T, int memLocation>
void Array_Flat<T, memLocation>::replace(int64_t index, const T& element)
{
  if(index < _size)
  {
    _data[index] = element;
  }
  else{
    std::cerr << "index = " << index << " while _size = " << _size << std::endl;
    assert(0);
  }
}

template <class T, int memLocation>
CUDA_CALLABLE const T& Array_Flat<T, memLocation>::operator[](int64_t index) const
{
   if ((index >= _size) || (index < 0)){
     assert(0); 
      //throw ArrayException("index is out of bounds");//CUDA does not support
   } 
   return _data[index];
}

template <class T, int memLocation>
CUDA_CALLABLE T& Array_Flat<T, memLocation>::operator[](int64_t index)
{
   if ((index >= _size) || (index < 0)){
     assert(0); 
      //throw ArrayException("index is out of bounds");//CUDA does not support
   } 
   return _data[index];
}

template <class T, int memLocation>
CUDA_CALLABLE Array_Flat<T, memLocation>& Array_Flat<T, memLocation>::operator=(const Array_Flat& rv)
{
   if (this == &rv) {
      return *this;
   }
   destructContents();
   copyContents(rv);
   return *this;
}

template <class T, int memLocation>
Array_Flat<T, memLocation>::~Array_Flat()
{
   destructContents();
}

/* 
 * IMPORTANT: has to be called after destructContents()
 *   to ensure _data is empty (no memory leak)
 *   Here:we assume the lvalue will allocate data on the same-memory-type (e.g. CPU or UnifiedMem) 
 *   with rvalue
 */
template <class T, int memLocation>
void Array_Flat<T, memLocation>::copyContents(const Array_Flat& rv)
{
   if (_size == 0 and rv._size == 0)
     return;
   //internalCopy(_data, rv._data); // multiplicating is done if necessary
   ////TUAN TODO: fix this as this is slow
   if (rv._size > 0)
   {
     if (_size > 0)
       destructContents();
     _mem_location = rv._mem_location;
     _array_design = rv._array_design;
     //_data = new T[sizeof(T) * _size];
     //_data = new((size_t)sizeof(T) * _size);
     //_data = (T*)new_memory((size_t)sizeof(T) * _size);
     //_allocated_size = _size;
     resize_allocated(rv._size, 0);
     _size = rv._size;
     //_incremental_size = rv._incremental_size; 
     //_communicatedSize = rv._communicatedSize;  
     //_sizeToCommunicate = rv._sizeToCommunicate;
     for (size_t j = 0; (j < _size); j++) {
       /* copy individual elements*/
       internalCopy(_data[j], rv._data[j]); // multiplicating is done if necessary
     };
   }
}

template <class T, int memLocation>
void Array_Flat<T, memLocation>::sort()
{
  size_t i;
  T temp;
  Array_Flat<T>& a=(*this);

  for (i=_size; i-- >0;)
    demote(i, _size-1);

  for (i=_size-1; i>=1; --i) {
    temp=a[0];
    a[0]=a[i];
    a[i]=temp;
    demote(0, i-1);
  }
}

template <class T, int memLocation>
void Array_Flat<T, memLocation>::demote(size_t boss, size_t bottomEmployee)
{
  Array_Flat<T>& a=(*this);
  size_t topEmployee;
  T temp;
  while (bottomEmployee>=2*boss) {
    topEmployee = 2*boss + ( ( (bottomEmployee!=2*boss) && (a[2*boss+1]>=a[2*boss] ) ) ? 1 : 0 );
    assert(topEmployee<_size);
    if (a[boss]<a[topEmployee]) {
      temp=a[boss];
      a[boss]=a[topEmployee];
      a[topEmployee]=temp;
      boss=topEmployee;
    }
    else break;
  }
}

template <class T, int memLocation>
void Array_Flat<T, memLocation>::unique()
{
  if (_size>0) {
    Array_Flat<T>& a=(*this);
    size_t i=0, j=0;
    while (j<_size) {
      while (j<_size && a[i]==a[j]) ++j;
      ++i;
      if (i<_size && j<_size) a[i]=a[j];
    }    
    decreaseSizeTo(i);
  }
}

template <class T, int memLocation>
void Array_Flat<T, memLocation>::merge(const Array_Flat& rv)
{
  size_t n=rv.size();
  if (n>0) {
    Array_Flat<T>& a=(*this);
    size_t m=_size;
    if ( m+n > std::numeric_limits<int64_t>::max())
    {
      std::cout << "ERROR: size too large\n";
      assert(0);
      return;
    }
    increaseSizeTo(m+n);
    //memcpy(&_data[_size-1], rv._data, n);
    std::copy_n(rv._data, n, &_data[_size-1]);
    //--m;
    //--n;
    //for (int i=_size-1; i>=0; --i) {
    //  if (m<0 || (n>=0 && rv[n]>a[m]) ) {
    //    a[i]=rv[n];
    //    --n;
    //  }
    //  else {
    //    a[i]=a[m];
    //    --m;
    //  }
    //}
  }
}
  
template <class T, int memLocation>
void Array_Flat<T, memLocation>::destructContents()
{
  if (_data != nullptr)
  {
    //delete _data;
    delete_memory(_data);
    _data = nullptr;
  }
  _size = 0;
  _allocated_size = 0;
}

/*
 * assign all n-elements of other array to the same value 'val'
 */
template <class T, int memLocation>
void Array_Flat<T, memLocation>::assign(int64_t n, const T& val)
{
  clear();
  increaseSizeTo(n);
  //TODO improve effiency
  for (size_t i = 0; i < n; ++i) {
    (*this)[i]=val;
  }
}

template <class T, int memLocation>
std::ostream& operator<<(std::ostream& os, const Array_Flat<T, memLocation>& arr) {
   size_t size = arr.size();
   for (size_t i = 0; i < size; ++i) {
      os << arr[i] << " ";
   }
   return os;
}

template <class T, int memLocation>
std::istream& operator>>(std::istream& is, Array_Flat<T, memLocation>& arr) {
//    unsigned size = arr.size();
//    for (unsigned i = 0; i < size; ++i) {
//       os << arr[i] << " ";
//    }
   assert(0);
   return is;
}

#endif

#endif
