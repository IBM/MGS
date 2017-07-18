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

#include "BValidOp.h"
#include "InFixOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include "PredicateException.h"
#include <memory>
#include <string>
#include <sstream>

void BValidOp::duplicate(std::auto_ptr<Operation>& rv) const
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


