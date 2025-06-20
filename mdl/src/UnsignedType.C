// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "UnsignedType.h"
#include "DataType.h"
#include <string>
#include <memory>

void UnsignedType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new UnsignedType(*this));
}

std::string UnsignedType::getDescriptor() const
{
   return "unsigned";
}

std::string UnsignedType::getCapitalDescriptor() const
{
   return "UnsignedInt";
}

bool UnsignedType::isBasic() const
{
   return true;
}

std::string UnsignedType::getArrayDataItemString() const
{
   return "IntArrayDataItem";
}

std::string UnsignedType::getDataItemString() const
{
   return getCapitalDescriptor() + "DataItem";
//   return "NumericDataItem";
}

std::string UnsignedType::getInitializationDataItemString() const
{
   return "NumericDataItem";
}

std::string UnsignedType::getDataItemFunctionString() const
{
   return "get" + getCapitalDescriptor() + "()";
}

std::string UnsignedType::getArrayInitializerString(const std::string& name
						, const std::string& arrayName
						, int level) const
{
   return getCustomArrayInitializerString(
      name, arrayName, level, "Int", "int");
}

UnsignedType::~UnsignedType() 
{
}
