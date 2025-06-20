// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GSValidOp.h"
#include "InFixOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include "PredicateException.h"
#include <memory>
#include <string>
#include <sstream>

void GSValidOp::duplicate(std::unique_ptr<Operation>&& rv) const
{
   rv.reset(new GSValidOp(*this));
}

std::string GSValidOp::getOp() const
{
   throw InternalException("GSValidOp::getOp is called.");
}

void GSValidOp::check(Predicate* p1, Predicate* p2) 
{
   std::ostringstream os;
   if ((classify(p1->getType()) == _Invalid) 
       || (classify(p1->getType()) == _Bool)) {
      os << p1->getType() << " is an invalid type for operation " << getOp();
      throw PredicateException(os.str());
   } 
   if ((classify(p2->getType()) == _Invalid) 
       || (classify(p2->getType()) == _Bool)) {
      os << p2->getType() << " is an invalid type for operation " << getOp();
      throw PredicateException(os.str());
   } 
   if (classify(p1->getType()) != classify(p2->getType())) {
      os << p1->getType() << " does not match " << p2->getType() 
	 << " for operation "  << getOp();
      throw PredicateException(os.str());
   } 
}

GSValidOp::~GSValidOp() 
{

}


