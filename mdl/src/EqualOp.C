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

#include "EqualOp.h"
#include "AllValidOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include "PredicateException.h"
#include <memory>
#include <string>
#include <sstream>

void EqualOp::duplicate(std::unique_ptr<Operation>&& rv) const
{
   rv.reset(new EqualOp(*this));
}

std::string EqualOp::getOp() const
{
   return "==";
}

EqualOp::~EqualOp() 
{

}


