// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "BValidOp.h"
#include "InFixOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include "PredicateException.h"
#include <memory>
#include <string>
#include <sstream>

void BValidOp::duplicate(std::unique_ptr<Operation>&& rv) const
{
   rv.reset(new BValidOp(*this));
}

std::string BValidOp::getOp() const
{
   throw InternalException("BValidOp::getOp is called.");
}

void BValidOp::check(Predicate* p1, Predicate* p2) 
{
   std::ostringstream os;
   if (classify(p1->getType()) != _Bool) {
      os << p1->getType() << " is an invalid type for operation " << getOp();
      throw PredicateException(os.str());
   } 
   if (classify(p2->getType()) != _Bool) {
      os << p2->getType() << " is an invalid type for operation " << getOp();
      throw PredicateException(os.str());
   } 
}

BValidOp::~BValidOp() 
{

}


