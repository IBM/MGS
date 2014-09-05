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
