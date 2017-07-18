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

   protected:
      virtual void internalSendRequest(int size) = 0;
      virtual void internalComplete() = 0;
      char* _buffer;
      int _bufferSize;
      int _peer;
      bool _active;
};

#endif
