// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include <cassert>
#include "AsynchReceiver.h"

AsynchReceiver::AsynchReceiver(char* buffer, int bufferSize, int peer)
   : _buffer(buffer), _bufferSize(bufferSize), _peer(peer), 
     _active(false)
{
}

void AsynchReceiver::receiveRequest() 
{
   assert (!_active); // this will change into an exception
   _active = true;
   internalReceiveRequest();
}

int AsynchReceiver::complete() 
{
   if (!_active) { // no communication to complete.
      return 0;
   }
   int retVal = internalComplete();
   _active = false;
   return retVal;	 
}

