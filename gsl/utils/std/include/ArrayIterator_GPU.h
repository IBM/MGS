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

#ifndef ArrayIterator_GPU_H
#define ArrayIterator_GPU_H
#include "Copyright.h"

#include <string>
#include <sstream>
#include <iterator>
#include <cstddef>

template<typename T, typename NonConstT>
class ArrayIterator
{
   public:
      // For Const iterators to reach non-const alikes internals.
      // Used for special conversion constructor below.
      friend class ArrayIterator<const T, T>;
      
      typedef ArrayIterator<T, NonConstT> self_type;

      typedef T& reference;
      typedef T* pointer;
      typedef int difference_type;
      typedef std::random_access_iterator_tag iterator_category;
      
      ArrayIterator() 
	 : _index(0), _currentData(nullptr) {}

      ArrayIterator(T** data_pointer, int index) 
	 : _data_pointer(data_pointer), _index(index), _currentData(nullptr) 
      {
	 setCurrentDataWithIndex();
      }

      // This is a constructor that creates const versions of non_const
      // iterators. It is necessary because begin() and end() return 
      // non-const iterators.
      ArrayIterator(
	 const ArrayIterator<NonConstT, NonConstT> &rv)
	 : _data_pointer(rv._data_pointer), 
	 _currentData(rv._currentData) {}

      reference operator*()  { 
	 return *_currentData;
      }

      pointer operator->() { 
	 return _currentData;
      }

      inline bool operator==(const ArrayIterator& rv) {
	 return (_data_pointer == rv._data_pointer) && (_currentData == rv._currentData);
      }

      inline bool operator!=(const ArrayIterator& rv) {
	 return !operator==(rv);
      }

      inline bool operator<=(const ArrayIterator& rv) {
	 return _index <= rv._index;
      }

      inline bool operator<(const ArrayIterator& rv) {
	 return _index < rv._index;
      }

      inline bool operator>=(const ArrayIterator& rv) {
	 return _index >= rv._index;
      }

      inline bool operator>(const ArrayIterator& rv) {
	 return _index > rv._index;
      }

      ArrayIterator operator+(unsigned int num) {
	 ArrayIterator retVal(_data_pointer, _index + num);
	 return retVal;
      }

      ArrayIterator& operator+=(unsigned int num) {
	 _index += num;
	 setCurrentDataWithIndex();
	 return *this;
      }

      /*
       * move per block
       */
      ArrayIterator operator-(unsigned int num) {
	 ArrayIterator retVal(_data_pointer, _index - num);
	 return retVal;
      }

      difference_type operator-(ArrayIterator& other) {
	 return _index - other._index;
      }

      ArrayIterator& operator-=(unsigned int num) {
	 _index -= num;
	 setCurrentDataWithIndex();
	 return *this;
      }

      ArrayIterator& operator++() {
	 _currentData++;
	 _index++;
	 return *this;
      }

      ArrayIterator operator++(int num) {
	 self_type retVal = *this;
	 ++(*this);
	 return retVal;
      }

      ArrayIterator& operator--() {
	 _currentData--;
	 _index--;
	 return *this;
      }

      ArrayIterator operator--(int num) {
	 self_type retVal = *this;
	 --(*this);
	 return retVal;
      }

   private:
      T** _data_pointer;
      int _index;
      T* _currentData;

      void setCurrentDataWithIndex() {
	 //_currentData = &((*_data_pointer)[_index]);      
	 if (*_data_pointer == nullptr)
	 {
	    _currentData == nullptr;
	 }
	 else{
	    _currentData = ((*_data_pointer) + _index);      
	 }
      }
};

#endif
