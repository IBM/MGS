// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "StringType.h"
#include "DataType.h"
#include "Constants.h"
#include <string>
#include <sstream>
#include <memory>
#include <vector>

void StringType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new StringType(*this));
}

std::string StringType::getDescriptor() const
{
   return "CustomString";
}

std::string StringType::getHeaderString(
   std::vector<std::string>& arrayTypeVec) const
{
   return "\"CustomString.h\"";
}

std::string StringType::getDataItemFunctionString() const
{
   return "getCustomString()";
}

StringType::~StringType() 
{
}
