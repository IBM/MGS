// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "ArrayException.h"

#ifndef Array_H
#define Array_H
#include "Copyright.h"

#include "ArrayException.h"
#include <cassert>
#include <memory>
#include <algorithm>

const unsigned SUGGESTEDARRAYBLOCKSIZE = 10;
const unsigned SUGGESTEDBLOCKINCREMENTSIZE = 4;

#define CUDA_OPTION_1  1    // used Managed class
#define CUDA_OPTION_2  2    // used custom allocator + Thrust
#define CUDA_OPTION  CUDA_OPTION_1

//#define USE_FLATARRAY_FOR_CONVENTIONAL_ARRAY

#if defined(HAVE_GPU) 
#include "ArrayIterator_GPU.h"
  #if CUDA_OPTION  == CUDA_OPTION_1
  #include "Array_GPU.h"
  #elif CUDA_OPTION  == CUDA_OPTION_2
  #include "Array_GPU_option2.h"

  #endif
#endif

#ifdef USE_FLATARRAY_FOR_CONVENTIONAL_ARRAY
//Only C++11
template <class T>
using Array = Array_Flat<T, 0>;

#else
#include "ArrayIterator.h"
template <class T>
class Array
{
 public:
    typedef Array<T> self_type;
    typedef T value_type;      
    typedef ArrayIterator<T, T> iterator;
    typedef ArrayIterator<const T, T> const_iterator;
    
    friend class ArrayIterator<T, T>;
    friend class ArrayIterator<const T, T>;

    Array(unsigned blockIncrementSize);
    /* delete all existing memory
     * then re-create to the minimal size
     * and set num-elements to 0
     */
    virtual void clear() {
      for (unsigned i = 0; i < _activeBlocks; i++) {
	delete[] _blocksArray[i];
      }
      delete[] _blocksArray;
      
      _size = 0;
      _activeBlocks = 0;
      
      _activeBlocksSize = getBlockIncrementSize();
      _blocksArray = new T*[_activeBlocksSize];
    }
    void increaseSizeTo(unsigned newSize);
    void decreaseSizeTo(unsigned newSize);
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
      return iterator(&_blocksArray, 0, getBlockSize());
    }
    
    iterator end() {
      return iterator(&_blocksArray, size(), getBlockSize());
    }
    
    const_iterator begin() const {
      return const_iterator(
			    const_cast<const T***>(&_blocksArray), 0, getBlockSize());
    }
    
    const_iterator end() const { 
      return const_iterator(
			    const_cast<const T***>(&_blocksArray), size(), getBlockSize());
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
	: _size(0), _communicatedSize(0),  _sizeToCommunicate(0), _activeBlocks(0), _activeBlocksSize(0), _blocksArray(0) {}

      virtual void internalCopy(T& lval, T& rval) = 0;
      virtual unsigned getBlockSize() const = 0;
      virtual unsigned getBlockIncrementSize() const = 0;
      /* wipe everything
       * don't create minimal data
       */
      void destructContents();
      void copyContents(const Array& rv); //have to be called after destructContents
      void demote (int, int); 
	  // NOTE: Arrays are organized in the form of multiple 'logical blocks'
	  //       i.e. memory increase/reduced, in the form of one or many blocks
      unsigned _size; //the number of elements in the array containing data
      unsigned _communicatedSize; //the number of elements MPI has communicated
      unsigned _sizeToCommunicate; //the number of elements MPI is to communicate
      unsigned _activeBlocks; //the number of active 
                      	  //blocks in the array (those really containing data)
      unsigned _activeBlocksSize;//the maximum number of elements that the allocated array
	                             // can hold
      T** _blocksArray; //the array holding the data
};

template <class T>
Array<T>::Array(unsigned blockIncrementSize)
   : _size(0), _communicatedSize(0), _sizeToCommunicate(0), _activeBlocks(0), _activeBlocksSize(0), _blocksArray(0)
{
   _activeBlocksSize += blockIncrementSize;
   _blocksArray = new T*[_activeBlocksSize];
}

/*
 * increase _size to 'newSize'
 * and if need, allocate more memory 
 */
template <class T>
void Array<T>::increaseSizeTo(unsigned newSize)
{
   while (_size < newSize) {
      increase();
   }
}

template <class T>
void Array<T>::decreaseSizeTo(unsigned newSize)
{
   while (_size > newSize) 
   {
      decrease();
   }
}

/*
 * increase _size 
 * and if need, allocate more memory 
 */
template <class T>
void Array<T>::increase()
{
   if ((_size % getBlockSize()) == 0) { // increase one block
     // The (_activeBlocks + 1) below is important it guarantees that
     // the iterator for end() will never segfault the high level array
     // is referenced in increase and constructor of the ArrayIterator.
     if (((_activeBlocks + 1) % getBlockIncrementSize()) == 0) { 
       // increase blocksArray 
       _activeBlocksSize += getBlockIncrementSize();
       T** tmp = new T*[_activeBlocksSize];
       for (unsigned i=0; i < (_activeBlocksSize - getBlockIncrementSize()); i++)
	 tmp[i]=_blocksArray[i];
       delete [] _blocksArray;
       _blocksArray = tmp;
     }
     _activeBlocks++;
     _blocksArray[_activeBlocks - 1] = new T[getBlockSize()];
   }
   _size++;
}

template <class T>
void Array<T>::decrease()
{
  --_size;
  if ((_size % getBlockSize()) == 0) {
    delete [] _blocksArray[_activeBlocks - 1];
    _blocksArray[_activeBlocks - 1]=0;
    --_activeBlocks;
    if (((_activeBlocks + 1) % getBlockIncrementSize()) == 0) {
      _activeBlocksSize -= getBlockIncrementSize();
      T** tmp = new T*[_activeBlocksSize];
      for (unsigned i=0; i < _activeBlocksSize; i++)
	tmp[i]=_blocksArray[i];
      delete [] _blocksArray;
      _blocksArray = tmp;
    }
  }
}

template <class T>
void Array<T>::insert(const T& element)
{
   increase();
   _blocksArray[_activeBlocks - 1][(_size - 1) % getBlockSize()] = element;
}

template <class T>
const T& Array<T>::operator[](int index) const
{
   if ((index >= _size) || (index < 0)){
      throw ArrayException("index is out of bounds");
   } 
   return _blocksArray[index / getBlockSize()][index % getBlockSize()];
}

template <class T>
T& Array<T>::operator[](int index)
{
   if ((index >= _size) || (index < 0)){
      throw ArrayException("index is out of bounds");
   } 
   return _blocksArray[index / getBlockSize()][index % getBlockSize()];
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

template <class T>
void Array<T>::copyContents(const Array& rv)
{
   _size = rv._size;
   _activeBlocks = rv._activeBlocks;
   _activeBlocksSize = rv._activeBlocksSize;
   _blocksArray = 0;
   if (rv._blocksArray) {
      unsigned curSize = 0;
      _blocksArray = new T*[_activeBlocksSize];
      for (unsigned i = 0; (i < _activeBlocks) && (curSize < _size); i++) {
	 _blocksArray[i] = new T[getBlockSize()];
	 for (unsigned j = 0; (j < getBlockSize()) && (curSize < _size); j++) {
	    internalCopy(_blocksArray[i][j], rv._blocksArray[i][j]); 
                                         // multiplicating is done if necessary
	    curSize++;
	 }
      }
   }
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
    --m;
    --n;
    for (int i=_size-1; i>=0; --i) {
      if (m<0 || (n>=0 && rv[n]>a[m]) ) {
	a[i]=rv[n];
	--n;
      }
      else {
	a[i]=a[m];
	--m;
      }
    }
  }
}
  
template <class T>
void Array<T>::destructContents()
{
   for (unsigned i = 0; i < _activeBlocks; i++) {
      delete[] _blocksArray[i];
   }
   delete[] _blocksArray;

   _blocksArray = 0;
   _size = 0;
   _activeBlocks = 0;
}

template <typename T>
void Array<T>::assign(unsigned n, const T& val)
{
  clear();
  increaseSizeTo(n);
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
