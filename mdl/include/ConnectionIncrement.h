// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef ConnectionIncrement_H
#define ConnectionIncrement_H
#include "Mdl.h"

#include <values.h>

class ConnectionIncrement
{
   public:
      ConnectionIncrement() {
         _computationTime = MAXFLOAT;
         _memoryBytes = 0;
         _communicationBytes = 0;
      }

      ConnectionIncrement(float ct, int mb, int cb) {
         _computationTime = ct;
         _memoryBytes = mb;
         _communicationBytes = cb;
      }

      void duplicate(std::auto_ptr<ConnectionIncrement>& rv) const
      {
         rv.reset(new ConnectionIncrement(*this));
      }

      inline ConnectionIncrement& operator= (ConnectionIncrement *c) {
         this->_computationTime = c->_computationTime;
         this->_memoryBytes = c->_memoryBytes;
         this->_communicationBytes = c->_communicationBytes;
         return *this;
      }

      inline ConnectionIncrement& operator= (ConnectionIncrement& c) {
         this->_computationTime = c._computationTime;
         this->_memoryBytes = c._memoryBytes;
         this->_communicationBytes = c._communicationBytes;
         return *this;
      }

      inline ConnectionIncrement& operator+= (ConnectionIncrement *c) {
         this->_computationTime += c->_computationTime;
         this->_memoryBytes += c->_memoryBytes;
         this->_communicationBytes += c->_communicationBytes;
         return *this;
      }

      inline ConnectionIncrement& operator+= (ConnectionIncrement& c) {
         this->_computationTime += c._computationTime;
         this->_memoryBytes += c._memoryBytes;
         this->_communicationBytes += c._communicationBytes;
         return *this;
      }

      float	_computationTime;
      int	_memoryBytes;
      int	_communicationBytes;
};

#endif
