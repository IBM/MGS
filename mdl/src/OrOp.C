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

#include "OrOp.h"
#include "BValidOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include "PredicateException.h"
#include <memory>
#include <string>
#include <sstream>

void OrOp::duplicate(std::auto_ptr<Operation>& rv) const
{
   rv.reset(new OrOp(*this));
}

std::string OrOp::getOp() const
{
   return "||";
}

OrOp::~OrOp() 
{

}


