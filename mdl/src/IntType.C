// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "IntType.h"
#include "SignedType.h"
#include "DataType.h"
#include <string>
#include <memory>

void IntType::duplicate(std::unique_ptr<DataType>&& rv) const
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
