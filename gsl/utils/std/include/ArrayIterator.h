// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef ArrayIterator_H
#define ArrayIterator_H
#include "Copyright.h"

#include <string>
#include <sstream>
#include <iterator>

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
	 : _blocksArray(0), _index(0), _currentData(0), _blockSize(0) {}

      ArrayIterator(T*** blocksArray, int index, int blockSize) 
	 : _blocksArray(blocksArray), _index(index), _currentData(0), 
	   _blockSize(blockSize) {
	 setCurrentDataWithIndex();
      }

      // This is a constructor that creates const versions of non_const
      // iterators. It is necessary because begin() and end() return 
      // non-const iterators.
      ArrayIterator(
	 const ArrayIterator<NonConstT, NonConstT> &rv)
	 : _blocksArray(rv._blocksArray), _index(rv._index),
	 _currentData(rv._currentData), _blockSize(rv._blockSize) {}

      reference operator*()  { 
	 return *_currentData;
      }

      pointer operator->() { 
	 return _currentData;
      }

      inline bool operator==(const ArrayIterator& rv) {
	 return (_blocksArray == rv._blocksArray) && (_index == rv._index) &&
	    (_currentData == rv._currentData);
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
	 ArrayIterator retVal(_blocksArray, _index + num, _blockSize);
	 return retVal;
      }

      ArrayIterator& operator+=(unsigned int num) {
	 _index += num;
	 setCurrentDataWithIndex();
	 return *this;
      }

      ArrayIterator operator-(unsigned int num) {
	 ArrayIterator retVal(_blocksArray, _index - num, _blockSize);
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
	 if (((_index + 1) % _blockSize) == 0) {
	    int blockIndex = (_index + 1) / _blockSize;
	    // it has to be the 0th element of the block.
	    _currentData = (*_blocksArray)[blockIndex]; 
	 } else {
	    _currentData++;
	 }
	 _index++;
	 return *this;
      }

      ArrayIterator operator++(int num) {
	 self_type retVal = *this;
	 ++(*this);
	 return retVal;
      }

      ArrayIterator& operator--() {
	 if ((_index % _blockSize) == 0) {
	    int blockIndex = (_index - 1) / _blockSize;
	    _currentData = (*_blocksArray)[blockIndex] + 
	       _blockSize - 1; 
	 } else {
	    _currentData--;
	 }
	 _index--;
	 return *this;
      }

      ArrayIterator operator--(int num) {
	 self_type retVal = *this;
	 --(*this);
	 return retVal;
      }

   private:
      T*** _blocksArray;
      int _index;
      T* _currentData;
      unsigned _blockSize;

      void setCurrentDataWithIndex() {
	 _currentData = (*_blocksArray)
	    [_index / _blockSize] + (_index % _blockSize);      
      }
};

#endif
