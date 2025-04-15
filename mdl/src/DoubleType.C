// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "DoubleType.h"
#include "DataType.h"
#include <string>
#include <memory>

void DoubleType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new DoubleType(*this));
}

std::string DoubleType::getDescriptor() const
{
   return "double";
}

bool DoubleType::isBasic() const
{
   return true;
}

std::string DoubleType::getArrayDataItemString() const
{
   return "FloatArrayDataItem";
}

std::string DoubleType::getCapitalDescriptor() const
{
   return "Double";
}

std::string DoubleType::getDataItemString() const
{
   return "DoubleDataItem";
}

std::string DoubleType::getInitializationDataItemString() const
{
   return "NumericDataItem";
}

std::string DoubleType::getDataItemFunctionString() const
{
   return "get" + getCapitalDescriptor() + "()";
}

std::string DoubleType::getArrayInitializerString(const std::string& name
						, const std::string& arrayName
						, int level) const
{
   return getCustomArrayInitializerString(
      name, arrayName, level, "Float", "float");
}

DoubleType::~DoubleType() 
{
}
