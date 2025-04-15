// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Predicate.h"
#include "PSetPredicate.h"
#include "StructType.h"
#include "Operation.h"
#include "InternalException.h"
#include <memory>
#include <string>
#include <iostream>

PSetPredicate::PSetPredicate() 
   : Predicate()
{
}

PSetPredicate::PSetPredicate(Operation* op, const std::string& name)
   : Predicate(op, name)
{
}

void PSetPredicate::duplicate(std::unique_ptr<Predicate>&& rv) const
{
   rv.reset(new PSetPredicate(*this));
}

void PSetPredicate::setPSet(StructType& type) 
{
   Predicate::setPSet(type);
   _type = getTypeUsingPSet(type);
   _name = type.getName() + "->" + _name;
}

std::string PSetPredicate::getTypeUsingPSet(StructType& type) 
{
   if(_name == "") {
      throw InternalException(
	 "_name is empty in PSetPredicate::getTypeUsingPSet");
   }
   DataType* data;
   // Connection should handle this to be more verbose about the error.
   try {
      data = type._members.getMember(_name);
   } catch (NotFoundException& e) {
      std::cerr << e.getError() << std::endl;
      e.setError("");
      throw;
   }
   return data->getDescriptor();
}

PSetPredicate::~PSetPredicate() 
{
}


