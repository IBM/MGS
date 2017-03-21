// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "InFixOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include <memory>
#include <string>
#include <sstream>

void InFixOp::duplicate(std::auto_ptr<Operation>& rv) const
{
   rv.reset(new InFixOp(*this));
}

void InFixOp::operate(Predicate* p1, Predicate* p2,  Predicate* cur) 
{
   std::ostringstream os;

   p1->operate();
   p2->operate();
   check(p1, p2);
   cur->setType("bool");
   os << p1->getName() << " " << getOp() << " " << p2->getName();
   cur->setName(os.str());
}

std::string InFixOp::getOp() const
{
   throw InternalException("InFixOp::getOp is called.");
}

void InFixOp::check(Predicate* p1, Predicate* p2) 
{
   throw InternalException("InFixOp::check is called.");
}

InFixOp::~InFixOp() 
{

}


