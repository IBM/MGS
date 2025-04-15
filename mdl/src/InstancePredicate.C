// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Predicate.h"
#include "InstancePredicate.h"
#include "StructType.h"
#include "Operation.h"
#include "InternalException.h"
#include <memory>
#include <string>
#include <iostream>

InstancePredicate::InstancePredicate() 
   : Predicate()
{
}

InstancePredicate::InstancePredicate(Operation* op, const std::string& name)
   : Predicate(op, name)
{
}

void InstancePredicate::duplicate(std::unique_ptr<Predicate>&& rv) const
{
   rv.reset(new InstancePredicate(*this));
}

void InstancePredicate::setInstances(const MemberContainer<DataType>& instances) 
{
   Predicate::setInstances(instances);
   _type = getTypeUsingInstance(instances);
}

std::string InstancePredicate::getTypeUsingInstance(
   const MemberContainer<DataType>& instances) 
{
   if(_name == "") {
      throw InternalException(
	 "_name is empty in InstancePredicate::getTypeUsingInstance");
   }
   const DataType* data;
   // Connection should handle this to be more verbose about the error.
   try {
      data = instances.getMember(_name);
   } catch (NotFoundException& e) {
      std::cerr << e.getError() << std::endl;
      e.setError("");
      throw;
   }
   return data->getDescriptor();
}

InstancePredicate::~InstancePredicate() 
{
}


