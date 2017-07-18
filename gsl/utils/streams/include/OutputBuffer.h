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
