// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Predicate.h"
#include "SharedPredicate.h"
#include "StructType.h"
#include "Operation.h"
#include "InternalException.h"
#include <memory>
#include <string>
#include <iostream>

SharedPredicate::SharedPredicate() 
   : Predicate()
{
}

SharedPredicate::SharedPredicate(Operation* op, const std::string& name)
   : Predicate(op, name)
{
}

void SharedPredicate::duplicate(std::unique_ptr<Predicate>&& rv) const
{
   rv.reset(new SharedPredicate(*this));
}

void SharedPredicate::setShareds(const MemberContainer<DataType>& shareds) 
{
   Predicate::setShareds(shareds);
   _type = getTypeUsingShared(shareds);
   _name = "getSharedMembers()." + _name;
}

std::string SharedPredicate::getTypeUsingShared(
   const MemberContainer<DataType>& shareds) 
{
   if(_name == "") {
      throw InternalException(
	 "_name is empty in SharedPredicate::getTypeUsingShared");
   }
   const DataType* data;
   // Connection should handle this to be more verbose about the error.
   try {
      data = shareds.getMember(_name);
   } catch (NotFoundException& e) {
      std::cerr << e.getError() << std::endl;
      e.setError("");
      throw;
   }
   return data->getDescriptor();
}

bool SharedPredicate::checkShareds()
{
   return true;
}

SharedPredicate::~SharedPredicate() 
{
}


