// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NotEqualOp.h"
#include "AllValidOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include "PredicateException.h"
#include <memory>
#include <string>
#include <sstream>

void NotEqualOp::duplicate(std::unique_ptr<Operation>&& rv) const
{
   rv.reset(new NotEqualOp(*this));
}

std::string NotEqualOp::getOp() const
{
   return "!=";
}

NotEqualOp::~NotEqualOp() 
{

}


