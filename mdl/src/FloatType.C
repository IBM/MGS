// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "FloatType.h"
#include "DataType.h"
#include <string>
#include <memory>

void FloatType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new FloatType(*this));
}

std::string FloatType::getDescriptor() const
{
   return "float";
}

bool FloatType::isBasic() const
{
   return true;
}

std::string FloatType::getArrayDataItemString() const
{
   return "FloatArrayDataItem";
}

std::string FloatType::getCapitalDescriptor() const
{
   return "Float";
}

std::string FloatType::getDataItemString() const
{
   return getCapitalDescriptor() + "DataItem";
}

std::string FloatType::getInitializationDataItemString() const
{
   return "NumericDataItem";
}

std::string FloatType::getDataItemFunctionString() const
{
   return "get" + getCapitalDescriptor() + "()";
}

std::string FloatType::getArrayInitializerString(const std::string& name
						, const std::string& arrayName
						, int level) const
{
   return getCustomArrayInitializerString(
      name, arrayName, level, "Float", "float");
}


FloatType::~FloatType() 
{
}
