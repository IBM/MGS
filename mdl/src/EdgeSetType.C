// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "EdgeSetType.h"
#include "LensType.h"
#include "DataType.h"
#include <string>
#include <memory>

void EdgeSetType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new EdgeSetType(*this));
}

std::string EdgeSetType::getDescriptor() const
{
   return "EdgeSet";
}

bool EdgeSetType::shouldBeOwned() const
{
   return true;
}

EdgeSetType::~EdgeSetType() 
{
}
