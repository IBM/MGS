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

#include "UnsignedType.h"
#include "DataType.h"
#include <string>
#include <memory>

void UnsignedType::duplicate(std::auto_ptr<DataType>& rv) const
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
