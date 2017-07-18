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

void PSetPredicate::duplicate(std::auto_ptr<Predicate>& rv) const
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


