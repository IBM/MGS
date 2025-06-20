// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "BoolType.h"
#include "DataType.h"
#include <string>
#include <memory>

void BoolType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new BoolType(*this));
}

std::string BoolType::getDescriptor() const
{
   return "bool";
}

std::string BoolType::getCapitalDescriptor() const
{
   return "Bool";
}

bool BoolType::isBasic() const
{
   return true;
}

std::string BoolType::getArrayDataItemString() const
{
   return "IntArrayDataItem";
}

std::string BoolType::getDataItemString() const
{
   return getCapitalDescriptor() + "DataItem";
//   return "NumericDataItem";
}

std::string BoolType::getInitializationDataItemString() const
{
   return "NumericDataItem";
}

std::string BoolType::getDataItemFunctionString() const
{
   return "get" + getCapitalDescriptor() + "()";
}

std::string BoolType::getArrayInitializerString(const std::string& name
						, const std::string& arrayName
						, int level) const
{
   return getCustomArrayInitializerString(
      name, arrayName, level, "Int", "int");
}

BoolType::~BoolType() 
{
}
