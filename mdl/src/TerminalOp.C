// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "TerminalOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include <memory>
#include <string>
#include <sstream>

void TerminalOp::duplicate(std::unique_ptr<Operation>&& rv) const
{
   rv.reset(new TerminalOp(*this));
}

void TerminalOp::operate(Predicate* p1, Predicate* p2,  Predicate* cur) 
{
}

TerminalOp::~TerminalOp() 
{

}


