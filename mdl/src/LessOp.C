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

#include "LessOp.h"
#include "GSValidOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include "PredicateException.h"
#include <memory>
#include <string>
#include <sstream>

void LessOp::duplicate(std::auto_ptr<Operation>& rv) const
{
   rv.reset(new LessOp(*this));
}

std::string LessOp::getOp() const
{
   return "<";
}

LessOp::~LessOp() 
{

}


