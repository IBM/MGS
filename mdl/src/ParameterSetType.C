// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ParameterSetType.h"
#include "LensType.h"
#include "DataType.h"
#include <string>
#include <memory>

void ParameterSetType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new ParameterSetType(*this));
}

std::string ParameterSetType::getDescriptor() const
{
   return "ParameterSet";
}

bool ParameterSetType::shouldBeOwned() const
{
   return true;
}

ParameterSetType::~ParameterSetType() 
{
}
