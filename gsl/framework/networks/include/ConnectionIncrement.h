// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ConnectionIncrement_H
#define ConnectionIncrement_H
#include "Copyright.h"

class ConnectionIncrement
{
   public:
      ConnectionIncrement() {
         _computationTime = 1;
         _memoryBytes = 4;
         _communicationBytes = 4;
         _mcomputationTime = 1;
         _mmemoryBytes = 4;
         _mcommunicationBytes = 4;
      }

      ConnectionIncrement(float ct, int mb, int cb, float mct, int mmb, int mcb) {
         _computationTime = ct;
         _memoryBytes = mb;
         _communicationBytes = cb;
         _mcomputationTime = mct;
         _mmemoryBytes = mmb;
         _mcommunicationBytes = mcb;
      }

      inline ConnectionIncrement& operator= (ConnectionIncrement *c) {
         this->_computationTime = c->_computationTime;
         this->_memoryBytes = c->_memoryBytes;
         this->_communicationBytes = c->_communicationBytes;
         this->_mcomputationTime = c->_mcomputationTime;
         this->_mmemoryBytes = c->_mmemoryBytes;
         this->_mcommunicationBytes = c->_mcommunicationBytes;
         return *this;
      }

      inline ConnectionIncrement& operator= (ConnectionIncrement& c) {
         this->_computationTime = c._computationTime;
         this->_memoryBytes = c._memoryBytes;
         this->_communicationBytes = c._communicationBytes;
         this->_mcomputationTime = c._mcomputationTime;
         this->_mmemoryBytes = c._mmemoryBytes;
         this->_mcommunicationBytes = c._mcommunicationBytes;
         return *this;
      }

      inline ConnectionIncrement& operator+= (ConnectionIncrement *c) {
         this->_computationTime += c->_computationTime;
         this->_memoryBytes += c->_memoryBytes;
         this->_communicationBytes += c->_communicationBytes;
         this->_mcomputationTime += c->_mcomputationTime;
         this->_mmemoryBytes += c->_mmemoryBytes;
         this->_mcommunicationBytes += c->_mcommunicationBytes;
         return *this;
      }

      inline ConnectionIncrement& operator+= (ConnectionIncrement& c) {
         this->_computationTime += c._computationTime;
         this->_memoryBytes += c._memoryBytes;
         this->_communicationBytes += c._communicationBytes;
         this->_mcomputationTime += c._mcomputationTime;
         this->_mmemoryBytes += c._mmemoryBytes;
         this->_mcommunicationBytes += c._mcommunicationBytes;
         return *this;
      }

      float	_computationTime;
      int	_memoryBytes;
      int	_communicationBytes;
      float	_mcomputationTime;     // marginal computation time
      int	_mmemoryBytes;         // marginal memory bytes
      int	_mcommunicationBytes;  // marginal communication bytes
};

#endif
