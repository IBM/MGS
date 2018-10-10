#ifndef ARRAY_GPU_H
#define ARRAY_GPU_H
#include <type_traits>
#include <algorithm>
#include <cstddef>
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


#ifndef CUDA_CHECK_CODE
#define CUDA_CHECK_CODE
#define CHECK(r) {_check((r), __LINE__);}

#endif

class Managed
{
  public:
    void _check(cudaError_t r, int line) {
      if (r != cudaSuccess) {
	printf("CUDA error on line %d: %s\n", line, cudaGetErrorString(r), line);
	exit(0);
      }
    }
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
      CHECK(cudaMallocManaged(&ptr, len));
      cudaDeviceSynchronize();
      return ptr;
    }
    void delete_memory(void* ptr)
    {
      cudaDeviceSynchronize();
      cudaFree(ptr);
    }
};

/*
 * CONSTRAINT: 
 *   to enable the iterrator to work properly, _data is at minimal 1 element
 */
template <class T>
class Array : public Managed
{
 public:
    typedef Array<T> self_type;
    typedef T value_type;      
    typedef ArrayIterator<T, T> iterator;
    typedef ArrayIterator<const T, T> const_iterator;
    
    friend class ArrayIterator<T, T>;
    friend class ArrayIterator<const T, T>;
    void * new_memory(size_t len)
    {
      /* testing: use again host allocation */
      void *ptr;
      ptr = ::new T[len/sizeof(T)];
      //cudaMallocManaged(&ptr, len);
      //cudaDeviceSynchronize();
      return ptr;
    }
    void delete_memory(T*& ptr)
    {
      //cudaDeviceSynchronize();
      //cudaFree(ptr);
      ::delete[] ptr;
    }

    //bool should_be_on_UM() {return true;};
    /* TUAN TODO probably safe to remove?
     * check if something likfe ShallowArray(a_number) or 
     * DeepPointerArray(a_number) is being used anywhere
     * YES: 
     *      DuplicatePointerArray<GranuleMapper, 50> _granuleMapperList;
     */
    Array(unsigned incrementSize);
    /* delete all existing memory
     * then re-create to the minimal size
     * and set num-elements to 0
     */
    virtual void clear() {
      if (_allocated_size > NUM_TRAILING_ELEMENT)
      {
	resize_allocated(NUM_TRAILING_ELEMENT, true);
      }
      _size = 0;
    }
    void increaseSizeTo(unsigned newSize);
    void decreaseSizeTo(unsigned newSize);
    //void resize_allocated(unsigned newSize);
    void resize_allocated(size_t newSize, bool force_trim_memory_to_smaller = false);
    void increase();
    void decrease();
    void assign(unsigned n, const T& val);
    void insert(const T& element);
    void push_back(const T& element) {
      insert(element);
    }
    T& operator[](int index);
    const T& operator[](int index) const;
    Array& operator=(const Array& rv);
    virtual void duplicate(std::unique_ptr<Array>& rv) const = 0;
    virtual ~Array();
    
    unsigned size() const {
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

    iterator begin() {
      return iterator(&_data, 0);
    }
    
    iterator end() {
      return iterator(&_data, size());
    }
    
    const_iterator begin() const {
      return const_iterator(const_cast<const T**>(&_data), 0);
    }
    
    const_iterator end() const { 
      return const_iterator(
			    const_cast<const T**>(&_data), size());
    }
     
    void sort();
    void unique();
    void merge(const Array& rv);

   protected:

      // This is required for the copy constructor of derived classes, 
      // the copy constructor of this class is really copyContents which uses
      // a virtual method that is implemented by the inheriting classes,
      // copyContents is also called by the inheriting classes
      Array()    
	: _allocated_size(0), _incremental_size(DEFAULT_INCREMENTAL_SIZE_ARR), 
	_size(0), _communicatedSize(0),  _sizeToCommunicate(0),
	_mem_location(MemLocation::CPU), _array_design(FLAT_ARRAY)
      {
	_data = nullptr;
	resize_allocated(MINIMAL_SIZE_ARR);
      }

      virtual void internalCopy(T& lval, T& rval) = 0;
      /* wipe everything
       * don't create minimal data
       */
      void destructContents();
      void copyContents(const Array& rv);
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
      typedef enum {CPU, UNIFIED_MEM} MemLocation;
      typedef enum {DOUBLE_ARRAY, FLAT_ARRAY} ArrayDesign;
};

template <class T>
Array<T>::Array(unsigned incrementSize)
  : _allocated_size(0), _incremental_size(incrementSize), _size(0), _communicatedSize(0),  _sizeToCommunicate(0)
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
 * just to make sure the allocated memory _allocated_size >=  'newSize'
 * IMPORTANT: do not change '_size'
 * if larger, maintain the existing data and physically allocate more if needed
 * if smaller, 
 *     (default)just trim the elements at the end [no change physically allocated memory]
 */
template <class T>
void Array<T>::resize_allocated(size_t newSize, bool force_trim_memory_to_smaller)
{
  if (newSize <= _allocated_size)
  {
    if (! force_trim_memory_to_smaller)
      return;
  }
  if (newSize <= 0)
  {
    assert(0);
    return;
  }
  int num_data_2_allocate = newSize;
  //T* new_data = new T[newSize*sizeof(T)];  // this use standard C++ 
  //T* new_data = new((size_t)newSize*sizeof(T));  // this use overloaded
  T* new_data = (T*)new_memory((size_t)num_data_2_allocate*sizeof(T));  // this use overloaded
  if (_data != nullptr)
  {
    if (_size > 0)
      std::copy_n((T*)_data, std::min(newSize, (size_t)_size), new_data);
    delete_memory(_data);  // this use overloadded 
  }
  _data = new_data;
  _allocated_size = num_data_2_allocate;
}

/*
 * make an array (i.e.i adjust _size) larger or smaller
 */
template <class T>
void Array<T>::increaseSizeTo(unsigned newSize)
{
  resize_allocated(newSize+NUM_TRAILING_ELEMENT);
  _size = newSize;
}


template <class T>
void Array<T>::decreaseSizeTo(unsigned newSize)
{
  resize_allocated(newSize+NUM_TRAILING_ELEMENT);
  _size = newSize;
}

/*
 * increase the logical value _size 1 element
 * and if needed, make more space
 */
template <class T>
void Array<T>::increase()
{
  resize_allocated(_size+1);
  _size += 1;
}

template <class T>
void Array<T>::decrease()
{
  resize_allocated(_size-1);
  _size -= 1;
}

template <class T>
void Array<T>::insert(const T& element)
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

template <class T>
const T& Array<T>::operator[](int index) const
{
   if ((index >= _size) || (index < 0)){
      throw ArrayException("index is out of bounds");
   } 
   return _data[index];
}

template <class T>
T& Array<T>::operator[](int index)
{
   if ((index >= _size) || (index < 0)){
      throw ArrayException("index is out of bounds");
   } 
   return _data[index];
}

template <class T>
Array<T>& Array<T>::operator=(const Array& rv)
{
   if (this == &rv) {
      return *this;
   }
   destructContents();
   copyContents(rv);
   return *this;
}

template <class T>
Array<T>::~Array()
{
   destructContents();
}

/* 
 * IMPORTANT: has to be called after destructContents()
 *   to ensure _data is empty (no memory leak)
 */
template <class T>
void Array<T>::copyContents(const Array& rv)
{
   _size = rv._size;
   //internalCopy(_data, rv._data); // multiplicating is done if necessary
   ////TUAN TODO: fix this as this is slow
   destructContents();
   //_data = new T[sizeof(T) * _size];
   //_data = new((size_t)sizeof(T) * _size);
   _data = (T*)new_memory((size_t)sizeof(T) * _size);
   _allocated_size = _size;
   for (unsigned j = 0; (j < _size); j++) {
     /* copy individual elements*/
     internalCopy(_data[j], rv._data[j]); // multiplicating is done if necessary
   };
}

template <class T>
void Array<T>::sort()
{
  int i;
  T temp;
  Array<T>& a=(*this);

  for (i=_size-1; i>=0; --i)
    demote(i, _size-1);

  for (i=_size-1; i>=1; --i) {
    temp=a[0];
    a[0]=a[i];
    a[i]=temp;
    demote(0, i-1);
  }
}

template <class T>
void Array<T>::demote(int boss, int bottomEmployee)
{
  Array<T>& a=(*this);
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

template <class T>
void Array<T>::unique()
{
  if (_size>0) {
    Array<T>& a=(*this);
    int i=0, j=0;
    while (j<_size) {
      while (j<_size && a[i]==a[j]) ++j;
      ++i;
      if (i<_size && j<_size) a[i]=a[j];
    }    
    decreaseSizeTo(i);
  }
}

template <class T>
void Array<T>::merge(const Array& rv)
{
  int n=rv.size();
  if (n>0) {
    Array<T>& a=(*this);
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
  
template <class T>
void Array<T>::destructContents()
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
template <typename T>
void Array<T>::assign(unsigned n, const T& val)
{
  clear();
  increaseSizeTo(n);
  //TODO improve effiency
  for (unsigned i = 0; i < n; ++i) {
    (*this)[i]=val;
  }
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Array<T>& arr) {
   unsigned size = arr.size();
   for (unsigned i = 0; i < size; ++i) {
      os << arr[i] << " ";
   }
   return os;
}

template<typename T>
std::istream& operator>>(std::istream& is, Array<T>& arr) {
//    unsigned size = arr.size();
//    for (unsigned i = 0; i < size; ++i) {
//       os << arr[i] << " ";
//    }
   assert(0);
   return is;
}

#endif

#endif
