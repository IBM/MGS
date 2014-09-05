// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "BoolType.h"
#include "DataType.h"
#include <string>
#include <memory>

void BoolType::duplicate(std::auto_ptr<DataType>& rv) const
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
