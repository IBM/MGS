// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef AsynchSender_H
#define AsynchSender_H
#include "Copyright.h"

class AsynchSender
{
   public:
      AsynchSender(char* buffer, int bufferSize, int peer);

      // Request an asynchronous transportation, should throw an exception
      // if there was once already without issuing a complete
      void sendRequest(int size);

      // Completes the transaction.
      virtual void complete();

      // Checks if the transaction is complete.
      virtual bool check() = 0;
      virtual ~AsynchSender() {};

   protected:
      virtual void internalSendRequest(int size) = 0;
      virtual void internalComplete() = 0;
      char* _buffer;
      int _bufferSize;
      int _peer;
      bool _active;
};

#endif
