#ifndef ARRAY_GPU_H
#define ARRAY_GPU_H
#include <type_traits>
#include <algorithm>
#include <cstddef>
#include "rndm.h"
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
    void * new_memory(size_t len)
    {
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
      ptr = ::new T[len/sizeof(T)];
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
      ::delete[] ptr;
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
    Array_Flat(unsigned incrementSize);
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
    void increaseSizeTo(unsigned newSize, bool force_trim_memory_to_smaller = false);
    void decreaseSizeTo(unsigned newSize);
    //void resize_allocated(unsigned newSize);
    void resize_allocated(size_t newSize, bool force_trim_memory_to_smaller = false);
    void resize_allocated_subarray(size_t MAX_SUBARRAY_SIZE, uint8_t location);
    void increase();
    void decrease();
    void assign(unsigned n, const T& val);
    void insert(const T& element);
    void push_back(const T& element) {
      insert(element);
    }
    CUDA_CALLABLE T& operator[](int index);
    CUDA_CALLABLE const T& operator[](int index) const;
    CUDA_CALLABLE Array_Flat& operator=(const Array_Flat& rv);
    //virtual void duplicate(std::unique_ptr<Array_Flat<T>>& rv) const = 0;
    virtual void duplicate(std::unique_ptr<Array_Flat<T, memLocation>>& rv) const = 0;
    virtual ~Array_Flat();
    
    CUDA_CALLABLE unsigned size() const {
      return _size;
    }
    
    const unsigned & getCommunicatedSize() const {
      return _communicatedSize;
    }

    unsigned & getCommunicatedSize() {
      return _communicatedSize;
    }

    void setCommunicatedSize(unsigned communicatedSize) {
      _communicatedSize = communicatedSize;
    }

    const unsigned & getSizeToCommunicate() const {
      return _sizeToCommunicate;
    }

    unsigned & getSizeToCommunicate() {
      return _sizeToCommunicate;
    }

    void setSizeToCommunicate(unsigned sizeToCommunicate) {
      _sizeToCommunicate = sizeToCommunicate;
    }

    CUDA_CALLABLE iterator begin() {
      return iterator(&_data, 0);
    }
    
    CUDA_CALLABLE iterator end() {
      return iterator(&_data, size());
    }
    
    CUDA_CALLABLE const_iterator begin() const {
      return const_iterator(const_cast<const T**>(&_data), 0);
    }
    
    CUDA_CALLABLE const_iterator end() const { 
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
	resize_allocated(MINIMAL_SIZE_ARR);
      }

      virtual void internalCopy(T& lval, T& rval) = 0;
      /* wipe everything
       * don't create minimal data
       */
      void destructContents();
      void copyContents(const Array_Flat& rv);
      void demote (int, int); 
	  // NOTE: Arrays are organized in the form of multiple 'logical blocks'
	  //       i.e. memory increase/reduced, in the form of one or many blocks
      int _allocated_size;  // the #elements as allocated
      int _incremental_size;  // the extra #elements to be iallocated
      unsigned _size; //the number of elements in the array containing data
      unsigned _communicatedSize; //the number of elements MPI has communicated
      unsigned _sizeToCommunicate; //the number of elements MPI is to communicate
      T* _data; // the flat array of data (to be on Unified Memory)
      uint8_t _mem_location;  //
      uint8_t _array_design;
   public:
      //TUAN: make this public so that we can use
      // Array_Flat<int>::MemLocation::CPU
      typedef enum {CPU=0, UNIFIED_MEM} MemLocation; //IMPORTANT: Keep CPU=0
      typedef enum {DOUBLE_ARRAY, FLAT_ARRAY} ArrayDesign;
};

template <class T, int memLocation>
Array_Flat<T, memLocation>::Array_Flat(unsigned incrementSize)
  : _allocated_size(0), _incremental_size(incrementSize), 
  _size(0), _communicatedSize(0),  _sizeToCommunicate(0),
  _mem_location(memLocation), _array_design(FLAT_ARRAY)
{
   if (_incremental_size == 0)
   {
     _incremental_size = DEFAULT_INCREMENTAL_SIZE_ARR; //default
   }
   _data = nullptr;
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
void Array_Flat<T, memLocation>::resize_allocated(size_t newSize, bool force_trim_memory_to_smaller)
{
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
  if ((int)newSize <= (int)_allocated_size-NUM_TRAILING_ELEMENT)
  {
    if (! force_trim_memory_to_smaller)
      return;
  }
  if (newSize < 0)
  {
    assert(0);
    return;
  }
  int num_data_2_allocate = newSize+NUM_TRAILING_ELEMENT;
  //T* new_data = new T[newSize*sizeof(T)];  // this use standard C++ 
  //T* new_data = new((size_t)newSize*sizeof(T));  // this use overloaded
  T* new_data = (T*)new_memory((size_t)num_data_2_allocate*sizeof(T));  // this use overloaded
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
      std::copy_n((T*)_data, std::min(newSize, (size_t)_size), new_data);
    delete_memory(_data);  // this use overloadded 
  }
  _data = new_data;
  _allocated_size = num_data_2_allocate;
}

template <class T, int memLocation>
void Array_Flat<T, memLocation>::resize_allocated_subarray(size_t MAX_SUBARRAY_SIZE,
   uint8_t mem_location)
{
//#define MAX_SUBARRAY_SIZE 20
  /* need to allocate memory */
  int newSize = MAX_SUBARRAY_SIZE;
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
void Array_Flat<T, memLocation>::increaseSizeTo(unsigned newSize, bool force_trim_memory_to_smaller)
{
  //resize_allocated(newSize+NUM_TRAILING_ELEMENT, force_trim_memory_to_smaller);
  resize_allocated(newSize, force_trim_memory_to_smaller);
  _size = newSize;
}


template <class T, int memLocation>
void Array_Flat<T, memLocation>::decreaseSizeTo(unsigned newSize)
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
    //_data = new T[_incremental_size*sizeof(T)];
    //_data = new((size_t)_incremental_size*sizeof(T));  // this use overloaded
    _data = (T*)new_memory((size_t)_incremental_size*sizeof(T));  // this use overloaded
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
CUDA_CALLABLE const T& Array_Flat<T, memLocation>::operator[](int index) const
{
   if ((index >= _size) || (index < 0)){
     assert(0); 
      //throw ArrayException("index is out of bounds");//CUDA does not support
   } 
   return _data[index];
}

template <class T, int memLocation>
CUDA_CALLABLE T& Array_Flat<T, memLocation>::operator[](int index)
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
     for (unsigned j = 0; (j < _size); j++) {
       /* copy individual elements*/
       internalCopy(_data[j], rv._data[j]); // multiplicating is done if necessary
     };
   }
}

template <class T, int memLocation>
void Array_Flat<T, memLocation>::sort()
{
  int i;
  T temp;
  Array_Flat<T>& a=(*this);

  for (i=_size-1; i>=0; --i)
    demote(i, _size-1);

  for (i=_size-1; i>=1; --i) {
    temp=a[0];
    a[0]=a[i];
    a[i]=temp;
    demote(0, i-1);
  }
}

template <class T, int memLocation>
void Array_Flat<T, memLocation>::demote(int boss, int bottomEmployee)
{
  Array_Flat<T>& a=(*this);
  int topEmployee;
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
    int i=0, j=0;
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
  int n=rv.size();
  if (n>0) {
    Array_Flat<T>& a=(*this);
    int m=_size;
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
void Array_Flat<T, memLocation>::assign(unsigned n, const T& val)
{
  clear();
  increaseSizeTo(n);
  //TODO improve effiency
  for (unsigned i = 0; i < n; ++i) {
    (*this)[i]=val;
  }
}

template <class T, int memLocation>
std::ostream& operator<<(std::ostream& os, const Array_Flat<T, memLocation>& arr) {
   unsigned size = arr.size();
   for (unsigned i = 0; i < size; ++i) {
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
