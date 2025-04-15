// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "SignedType.h"
#include "DataType.h"
#include <string>
#include <memory>

SignedType::SignedType() 
   : DataType(), _signed(true) 
{

}

void SignedType::duplicate(std::unique_ptr<DataType>&& rv) const
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
