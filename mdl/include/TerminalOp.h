// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TerminalOp_H
#define TerminalOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"

class Predicate;

class TerminalOp : public Operation {

   public:
      virtual void duplicate(std::unique_ptr<Operation>&& rv) const;
      virtual ~TerminalOp();
      virtual void operate(Predicate* p1, Predicate* p2,  Predicate* cur);
};


#endif // TerminalOp_H
