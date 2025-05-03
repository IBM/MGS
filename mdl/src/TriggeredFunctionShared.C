// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include <memory>
#include "TriggeredFunctionShared.h"
#include "Constants.h"

TriggeredFunctionShared::TriggeredFunctionShared(
   const std::string& name, RunType runType)
   : TriggeredFunction(name, runType)
{
}

void TriggeredFunctionShared::duplicate(
   std::unique_ptr<TriggeredFunction>&& rv) const
{
   rv.reset(new TriggeredFunctionShared(*this));
}

TriggeredFunctionShared::~TriggeredFunctionShared()
{
}

std::string TriggeredFunctionShared::getTriggerableType (
   const std::string& modelName) const
{   
   return modelName + COMPCATEGORY;
}
