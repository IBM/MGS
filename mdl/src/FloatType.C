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

#include "FloatType.h"
#include "DataType.h"
#include <string>
#include <memory>

void FloatType::duplicate(std::auto_ptr<DataType>& rv) const
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
