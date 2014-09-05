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
