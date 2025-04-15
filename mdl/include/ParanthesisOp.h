// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ParanthesisOp_H
#define ParanthesisOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"

class Predicate;

class ParanthesisOp : public Operation {

   public:
      virtual void duplicate(std::unique_ptr<Operation>&& rv) const;
      virtual ~ParanthesisOp();
      virtual void operate(Predicate* p1, Predicate* p2, Predicate* cur);
};


#endif // ParanthesisOp_H
