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

#include "GSValidOp.h"
#include "InFixOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include "PredicateException.h"
#include <memory>
#include <string>
#include <sstream>

void GSValidOp::duplicate(std::auto_ptr<Operation>& rv) const
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


