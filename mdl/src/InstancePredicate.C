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


