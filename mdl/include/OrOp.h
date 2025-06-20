// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef OrOp_H
#define OrOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"
#include "BValidOp.h"

class Predicate;

class OrOp : public BValidOp {

   public:
      virtual void duplicate(std::unique_ptr<Operation>&& rv) const;
      virtual ~OrOp();
      virtual std::string getOp() const;

};


#endif // OrOp_H
