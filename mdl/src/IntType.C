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

#include "IntType.h"
#include "SignedType.h"
#include "DataType.h"
#include <string>
#include <memory>

void IntType::duplicate(std::auto_ptr<DataType>& rv) const
{
   rv.reset(new IntType(*this));
}

std::string IntType::getDescriptor() const
{
   return "int";
}

std::string IntType::getCapitalDescriptor() const
{
   return "Int";
}

IntType::~IntType() 
{
}
