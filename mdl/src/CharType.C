// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "CharType.h"
#include "SignedType.h"
#include "DataType.h"
#include <string>
#include <memory>

void CharType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new CharType(*this));
}

std::string CharType::getDescriptor() const
{
   return "char";
}

std::string CharType::getCapitalDescriptor() const
{
   return "Char";
}

CharType::~CharType() 
{
}
