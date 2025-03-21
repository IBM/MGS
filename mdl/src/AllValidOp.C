// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "AllValidOp.h"
#include "InFixOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include "PredicateException.h"
#include <memory>
#include <string>
#include <sstream>

void AllValidOp::duplicate(std::unique_ptr<Operation>&& rv) const
{
   rv.reset(new AllValidOp(*this));
}

std::string AllValidOp::getOp() const
{
   throw InternalException("AllValidOp::getOp is called.");
}

void AllValidOp::check(Predicate* p1, Predicate* p2) 
{
   std::ostringstream os;
   if (classify(p1->getType()) == _Invalid) {
      os << p1->getType() << " is an invalid type for operation " << getOp();
      throw PredicateException(os.str());
   } 
   if (classify(p2->getType()) == _Invalid) {
      os << p2->getType() << " is an invalid type for operation " << getOp();
      throw PredicateException(os.str());
   } 
   if (classify(p1->getType()) != classify(p2->getType())) {
      os << p1->getType() << " does not match " << p2->getType() 
	 << " for operation "  << getOp();
      throw PredicateException(os.str());
   } 
}

AllValidOp::~AllValidOp() 
{

}


