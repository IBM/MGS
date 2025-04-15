// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NotEqualOp_H
#define NotEqualOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"
#include "AllValidOp.h"

class Predicate;

class NotEqualOp : public AllValidOp {

   public:
      virtual void duplicate(std::unique_ptr<Operation>&& rv) const;
      virtual ~NotEqualOp();
      virtual std::string getOp() const;

};


#endif // NotEqualOp_H
