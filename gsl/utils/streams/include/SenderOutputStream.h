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

#ifndef SenderOutputStream_H
#define SenderOutputStream_H
#include "Copyright.h"

#include "OutputBuffer.h"
#include "OutputStream.h"
#include "StreamConstants.h"
#include <string>
#include <iostream>

class Simulation;

template <typename SenderType>
class SenderOutputStream : public OutputStream {
  public:
  SenderOutputStream(int peer, int tag, Simulation* sim);
  virtual ~SenderOutputStream();

  void flush();
  virtual void reset();

  protected:
  virtual inline void write(const char* data, int size);

  inline int nextBuffer() const { return (_current + 1) % 2; }
  int _current;
  OutputBuffer<BUFFERSIZE>* _buffers[2];
  SenderType* _senders[2];
};

template <typename SenderType>
SenderOutputStream<SenderType>::SenderOutputStream(int peer, int tag,
                                                   Simulation* sim)
    : _current(0) {

  _buffers[0] = new OutputBuffer<BUFFERSIZE>;
  _buffers[1] = new OutputBuffer<BUFFERSIZE>;
  _senders[0] = new SenderType(_buffers[0]->_data, BUFFERSIZE, peer, tag, sim);
  _senders[1] = new SenderType(_buffers[1]->_data, BUFFERSIZE, peer, tag, sim);
}

template <typename SenderType>
SenderOutputStream<SenderType>::~SenderOutputStream() {
  flush();
  for (int i = 0; i < 2; ++i) {
    if (!_buffers[i]->_ready) {
      _senders[i]->complete();
    }
  }
  delete _buffers[0];
  delete _buffers[1];
  delete _senders[0];
  delete _senders[1];
}

template <typename SenderType>
void SenderOutputStream<SenderType>::flush() {
  if (_buffers[_current]->_index != 0) {
    _buffers[_current]->_ready = false;
    _senders[_current]->sendRequest(
        _buffers[_current]->_index);  // this sender must not be active
    _buffers[_current]->_index = 0;
    _current = nextBuffer();
  }
}

template <typename SenderType>
void SenderOutputStream<SenderType>::reset() {
  if (_buffers[_current]->_ready) {
    flush();
  }
}

template <typename SenderType>
void SenderOutputStream<SenderType>::write(const char* data, int size) {
  int totalWriteSize = 0, writeSize, remainingSize;
  do {
    if (_buffers[_current]->_ready) {
      remainingSize = size - totalWriteSize;
      if (_buffers[_current]->remainingSize() > remainingSize) {
        std::copy(data + totalWriteSize, data + totalWriteSize + remainingSize,
                  _buffers[_current]->currentPositionAndUpdate(remainingSize));
        // memcpy(_buffers[_current]->currentPositionAndUpdate(remainingSize),
        //        data + totalWriteSize, remainingSize);
        break;
      } else {
        writeSize = _buffers[_current]->remainingSize();
        std::copy(data + totalWriteSize, data + totalWriteSize + writeSize,
                  _buffers[_current]->currentPositionAndUpdate(writeSize));
        // memcpy(_buffers[_current]->currentPositionAndUpdate(writeSize),
        //       data + totalWriteSize, writeSize);
        totalWriteSize += writeSize;
        flush();
      }
    } else {                           // not ready after a flush
      _senders[_current]->complete();  // this must be called to make the sender
                                       // inactive and prepare for flush
      _buffers[_current]->_ready = true;
    }
  } while (totalWriteSize < size);
}

#endif
