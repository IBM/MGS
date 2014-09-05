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

#ifndef AsynchReceiver_H
#define AsynchReceiver_H
#include "Copyright.h"

class AsynchReceiver
{
   public:
      AsynchReceiver(char* buffer, int bufferSize, int peer);

      // Request an asynchronous transportation, should throw an exception
      // if there was once already without issuing a complete
      void receiveRequest(); 

      // Completes the transaction, returns the number of chars received.
      virtual int complete();

      // Checks if the transaction is complete.
      virtual bool check() = 0;
   protected:
      virtual void internalReceiveRequest() = 0;
      virtual int internalComplete() = 0;
      char* _buffer;
      int _bufferSize;
      int _peer;
      bool _active;
};

#endif
