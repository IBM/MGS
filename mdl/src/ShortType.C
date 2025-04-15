// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ShortType.h"
#include "SignedType.h"
#include "DataType.h"
#include <string>
#include <memory>

void ShortType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new ShortType(*this));
}

std::string ShortType::getDescriptor() const
{
   return "short";
}

std::string ShortType::getCapitalDescriptor() const
{
   return "Short";
}

ShortType::~ShortType() 
{
}
