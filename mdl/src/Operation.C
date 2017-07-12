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

#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include <memory>
#include <string>

void Operation::duplicate(std::auto_ptr<Operation>& rv) const
{
   rv.reset(new Operation(*this));
}

void Operation::operate(Predicate* p1, Predicate* p2, Predicate* cur) 
{
   throw InternalException("Operation::operate is called.");   
}

Operation::_Type Operation::classify(std::string s) 
{
   if ((s == "string") || (s == "String")) {
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


