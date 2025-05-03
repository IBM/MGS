// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include <memory>
#ifndef ConnectionIncrement_H
#define ConnectionIncrement_H
#include "Mdl.h"
#include <float.h>

class ConnectionIncrement
{
   public:
      ConnectionIncrement() {
         _computationTime = FLT_MAX;
         _memoryBytes = 0;
         _communicationBytes = 0;
      }

      ConnectionIncrement(float ct, int mb, int cb) {
         _computationTime = ct;
         _memoryBytes = mb;
         _communicationBytes = cb;
      }

      void duplicate(std::unique_ptr<ConnectionIncrement>&& rv) const
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
