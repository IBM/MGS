// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ParanthesisOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include <memory>
#include <string>
#include <sstream>

void ParanthesisOp::duplicate(std::unique_ptr<Operation>&& rv) const
{
   rv.reset(new ParanthesisOp(*this));
}

void ParanthesisOp::operate(Predicate* p1, Predicate* p2,  Predicate* cur) 
{
   std::ostringstream os;

   p1->operate();
   cur->setType(p1->getType());
   os << "( " << p1->getName() << " )";
   cur->setName(os.str());
}

ParanthesisOp::~ParanthesisOp() {

}


