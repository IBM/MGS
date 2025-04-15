// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include <cassert>
#include "AsynchSender.h"

AsynchSender::AsynchSender(char* buffer, int bufferSize, int peer)
   : _buffer(buffer), _bufferSize(bufferSize), _peer(peer), 
     _active(false)
{
}

void AsynchSender::sendRequest(int size) 
{
   //assert (!_active); // this will change into an exception
   _active = true;
   internalSendRequest(size);
}

void AsynchSender::complete() 
{
  if (_active) { // if not active, no communication to complete
    internalComplete();
    _active = false;
  }
}
