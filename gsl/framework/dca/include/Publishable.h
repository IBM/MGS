// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PUBLISHABLE_H
#define PUBLISHABLE_H
//#include "Copyright.h"


class Publisher;

class Publishable
{
   protected:
      //      Publisher* _ptrPublisher;
   public:
      //      Publishable() : _ptrPublisher(0) {}
      virtual Publisher* getPublisher() =0;
      virtual ~Publishable() {}
      virtual const char* getServiceName(void* data) const = 0;
      virtual const char* getServiceDescription(void* data) const = 0;
};
#endif
