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

void SharedPredicate::duplicate(std::auto_ptr<Predicate>& rv) const
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


