// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include <memory>
#include "TriggeredFunctionInstance.h"

TriggeredFunctionInstance::TriggeredFunctionInstance(
   const std::string& name, RunType runType)
   : TriggeredFunction(name, runType)
{
}

void TriggeredFunctionInstance::duplicate(
   std::unique_ptr<TriggeredFunction>&& rv) const
{
   rv.reset(new TriggeredFunctionInstance(*this));
}

TriggeredFunctionInstance::~TriggeredFunctionInstance()
{
}

std::string TriggeredFunctionInstance::getTriggerableType (
   const std::string& modelName) const
{
   return modelName;
}
