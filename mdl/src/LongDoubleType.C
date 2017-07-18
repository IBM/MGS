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

#include "LongDoubleType.h"
#include "DataType.h"
#include <string>
#include <memory>

void LongDoubleType::duplicate(std::auto_ptr<DataType>& rv) const
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
