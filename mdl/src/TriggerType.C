// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "TriggerType.h"
#include "LensType.h"
#include "DataType.h"
#include <string>
#include <memory>

void TriggerType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new TriggerType(*this));
}

std::string TriggerType::getDescriptor() const
{
   return "Trigger";
}

TriggerType::~TriggerType() 
{
}
