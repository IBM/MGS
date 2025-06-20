// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef OutputBuffer_H
#define OutputBuffer_H
#include "Copyright.h"

template<int _bufferSize>
struct OutputBuffer
{      
      OutputBuffer() 
	 : _index(0), _ready(true)
	 {}
      inline int remainingSize() const {
	 return _bufferSize - _index;
      }
      inline char* currentPositionAndUpdate(int size) {
	 char* retval = &_data[_index];
	 _index += size;
	 return retval;
      }
      int _index;
      bool _ready;
      char _data[_bufferSize];
};

#endif
