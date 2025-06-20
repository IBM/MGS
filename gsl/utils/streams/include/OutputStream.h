// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
  OutputStream() : _buffer(0), _bufferPtr(0), _rebuild(false) {}
  OutputStream(char* buffer) : _buffer(buffer), _bufferPtr(buffer), _rebuild(false) {}
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

  virtual void reset() {
    _bufferPtr = _buffer;
    _rebuild = false;
  }

  void requestRebuild(bool rebuild) { _rebuild = rebuild; }
  bool rebuildRequested() { return _rebuild; }

  protected:
  virtual inline void write(const char* data, int size) {
    // std::memcpy(_bufferPtr, data, size);
    std::copy(data, data + size, _bufferPtr);
    _bufferPtr += size;
  }

  private:
  char* _buffer;
  char* _bufferPtr;
  bool _rebuild;
};

#endif
