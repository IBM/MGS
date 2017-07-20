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

#include "DoubleType.h"
#include "DataType.h"
#include <string>
#include <memory>

void DoubleType::duplicate(std::auto_ptr<DataType>& rv) const
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
