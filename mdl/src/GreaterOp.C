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

#include "GreaterOp.h"
#include "GSValidOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include "PredicateException.h"
#include <memory>
#include <string>
#include <sstream>

void GreaterOp::duplicate(std::auto_ptr<Operation>& rv) const
{
   rv.reset(new GreaterOp(*this));
}

std::string GreaterOp::getOp() const
{
   return ">";
}

GreaterOp::~GreaterOp() 
{

}


