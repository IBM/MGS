// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "FunctorType.h"
#include "GslType.h"
#include <string>
#include "DataType.h"
#include <memory>

void FunctorType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new FunctorType(*this));
}

std::string FunctorType::getDescriptor() const
{
   return "Functor";
}

bool FunctorType::shouldBeOwned() const
{
   return true;
}

FunctorType::~FunctorType() 
{
}
