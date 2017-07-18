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

#ifndef OutputStream_H
#define OutputStream_H
#include "Copyright.h"

#include "OutputBuffer.h"
#include "StreamConstants.h"
//#include <cstring>
#include <algorithm>
#include <iostream>

class OutputStream {
  public:
  OutputStream() {}
  OutputStream(char* buffer) : _buffer(buffer), _bufferPtr(buffer) {}
  virtual ~OutputStream() {}

  template <typename T>
  OutputStream& operator<<(T& data) {
    write((const char*)&data, sizeof(T));
    return *this;
  }

  OutputStream& operator<<(std::string& data) {
    int size = data.size();
    write((const char*)&size, sizeof(int));
    write(data.c_str(), size);
    return *this;
  }

  virtual void reset() { _bufferPtr = _buffer; }

  protected:
  virtual inline void write(const char* data, int size) {
    // std::memcpy(_bufferPtr, data, size);
    std::copy(data, data + size, _bufferPtr);
    _bufferPtr += size;
  }

  private:
  char* _buffer;
  char* _bufferPtr;
};

#endif
