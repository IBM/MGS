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

#include "SignedType.h"
#include "DataType.h"
#include <string>
#include <memory>

SignedType::SignedType() 
   : DataType(), _signed(true) 
{

}

void SignedType::duplicate(std::auto_ptr<DataType>& rv) const
{
   rv.reset(new SignedType(*this));
}

bool SignedType::isSigned() const
{
   return _signed;
}

void SignedType::setSigned(bool sign) 
{
   _signed = sign;
}

std::string SignedType::getDescriptor() const
{
   return "SignedTypeBaseClass";
}

bool SignedType::isBasic() const
{
   return true;
}

std::string SignedType::getDataItemString() const
{
   return getCapitalDescriptor() + "DataItem";
}

std::string SignedType::getInitializationDataItemString() const
{
   return "NumericDataItem";
}

std::string SignedType::getArrayDataItemString() const
{
   return "IntArrayDataItem";
}

std::string SignedType::getDataItemFunctionString() const
{
   return "get" + getCapitalDescriptor() + "()";
}

std::string SignedType::getArrayInitializerString(const std::string& name
						, const std::string& arrayName
						, int level) const
{
   return getCustomArrayInitializerString(name, arrayName
					  , level, "Int", "int");
}

SignedType::~SignedType() 
{
}
