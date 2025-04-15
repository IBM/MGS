// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "LongDoubleType.h"
#include "DataType.h"
#include <string>
#include <memory>

void LongDoubleType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new LongDoubleType(*this));
}

std::string LongDoubleType::getDescriptor() const
{
   return "long double";
}

bool LongDoubleType::isBasic() const
{
   return true;
}

std::string LongDoubleType::getCapitalDescriptor() const
{
   return "Double";
}

std::string LongDoubleType::getArrayDataItemString() const
{
   return "DoubleArrayDataItem";
}

std::string LongDoubleType::getDataItemString() const
{
   return getCapitalDescriptor() + "DataItem";
//   return "NumericDataItem";
}

std::string LongDoubleType::getInitializationDataItemString() const
{
   return "NumericDataItem";
}

std::string LongDoubleType::getDataItemFunctionString() const
{
   return "get" + getCapitalDescriptor() + "()";
}

std::string LongDoubleType::getArrayInitializerString(const std::string& name
						, const std::string& arrayName
						, int level) const
{
   return getCustomArrayInitializerString(name, arrayName, level, "Float"
					  , "float");
}

LongDoubleType::~LongDoubleType() 
{
}
