// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include <memory>
#include <string>

void Operation::duplicate(std::unique_ptr<Operation>&& rv) const
{
   rv.reset(new Operation(*this));
}

void Operation::operate(Predicate* p1, Predicate* p2, Predicate* cur) 
{
   throw InternalException("Operation::operate is called.");   
}

Operation::_Type Operation::classify(std::string s) 
{
   if ((s == "string") || (s == "CustomString")) {
      return _String;
   } else if (s == "bool") {
      return _Bool;
   } else if ((s == "char") || (s == "short") || (s == "int") || (s == "long") 
	      || (s == "float") || (s == "double") || (s == "long double") 
	      || (s == "unsigned") ) {
      return _General;
   } else {
      return _Invalid;
   }
}

Operation::~Operation() 
{

}


