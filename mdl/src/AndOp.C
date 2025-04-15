// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "AndOp.h"
#include "BValidOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include "PredicateException.h"
#include <memory>
#include <string>
#include <sstream>

void AndOp::duplicate(std::unique_ptr<Operation>&& rv) const
{
   rv.reset(new AndOp(*this));
}

std::string AndOp::getOp() const
{
   return "&&";
}

AndOp::~AndOp() 
{

}


