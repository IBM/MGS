// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ArrayIterator_GPU_H
#define ArrayIterator_GPU_H
#include "Copyright.h"

#include <string>
#include <sstream>
#include <iterator>
#include <cstddef>
#include "rndm.h"

template<typename T, typename NonConstT>
class Array_FlatIterator
{
   public:
      // For Const iterators to reach non-const alikes internals.
      // Used for special conversion constructor below.
      friend class Array_FlatIterator<const T, T>;
      
      typedef Array_FlatIterator<T, NonConstT> self_type;

      typedef T& reference;
      typedef T* pointer;
      typedef int difference_type;
      typedef std::random_access_iterator_tag iterator_category;
      
      CUDA_CALLABLE Array_FlatIterator() 
	 : _index(0), _currentData(nullptr) {}

      CUDA_CALLABLE Array_FlatIterator(T** data_pointer, int index) 
	 : _data_pointer(data_pointer), _index(index), _currentData(nullptr) 
      {
	 setCurrentDataWithIndex();
      }

      // This is a constructor that creates const versions of non_const
      // iterators. It is necessary because begin() and end() return 
      // non-const iterators.
      CUDA_CALLABLE Array_FlatIterator(
	 const Array_FlatIterator<NonConstT, NonConstT> &rv)
	 : _data_pointer(rv._data_pointer), 
	 _currentData(rv._currentData) {}

      CUDA_CALLABLE reference operator*()  { 
	 return *_currentData;
      }

      CUDA_CALLABLE pointer operator->() { 
	 return _currentData;
      }

      CUDA_CALLABLE inline bool operator==(const Array_FlatIterator& rv) {
	 return (_data_pointer == rv._data_pointer) && (_currentData == rv._currentData);
      }

      CUDA_CALLABLE inline bool operator!=(const Array_FlatIterator& rv) {
	 return !operator==(rv);
      }

      CUDA_CALLABLE inline bool operator<=(const Array_FlatIterator& rv) {
	 return _index <= rv._index;
      }

      CUDA_CALLABLE inline bool operator<(const Array_FlatIterator& rv) {
	 return _index < rv._index;
      }

      CUDA_CALLABLE inline bool operator>=(const Array_FlatIterator& rv) {
	 return _index >= rv._index;
      }

      CUDA_CALLABLE inline bool operator>(const Array_FlatIterator& rv) {
	 return _index > rv._index;
      }

      CUDA_CALLABLE Array_FlatIterator operator+(unsigned int num) {
	 Array_FlatIterator retVal(_data_pointer, _index + num);
	 return retVal;
      }

      CUDA_CALLABLE Array_FlatIterator& operator+=(unsigned int num) {
	 _index += num;
	 setCurrentDataWithIndex();
	 return *this;
      }

      /*
       * move per block
       */
      CUDA_CALLABLE Array_FlatIterator operator-(unsigned int num) {
	 Array_FlatIterator retVal(_data_pointer, _index - num);
	 return retVal;
      }

      CUDA_CALLABLE difference_type operator-(Array_FlatIterator& other) {
	 return _index - other._index;
      }

      CUDA_CALLABLE Array_FlatIterator& operator-=(unsigned int num) {
	 _index -= num;
	 setCurrentDataWithIndex();
	 return *this;
      }

      CUDA_CALLABLE Array_FlatIterator& operator++() {
	 _currentData++;
	 _index++;
	 return *this;
      }

      CUDA_CALLABLE Array_FlatIterator operator++(int num) {
	 self_type retVal = *this;
	 ++(*this);
	 return retVal;
      }

      CUDA_CALLABLE Array_FlatIterator& operator--() {
	 _currentData--;
	 _index--;
	 return *this;
      }

      CUDA_CALLABLE Array_FlatIterator operator--(int num) {
	 self_type retVal = *this;
	 --(*this);
	 return retVal;
      }

   private:
      T** _data_pointer;
      int _index;
      T* _currentData;

      CUDA_CALLABLE void setCurrentDataWithIndex() {
	 //_currentData = &((*_data_pointer)[_index]);      
	 if (*_data_pointer == nullptr)
	 {
	    _currentData = nullptr;
	 }
	 else{
	    _currentData = ((*_data_pointer) + _index);      
	 }
      }
};

#endif
