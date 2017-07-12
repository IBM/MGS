// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "TriggeredFunctionInstance.h"

TriggeredFunctionInstance::TriggeredFunctionInstance(
   const std::string& name, RunType runType)
   : TriggeredFunction(name, runType)
{
}

void TriggeredFunctionInstance::duplicate(
   std::auto_ptr<TriggeredFunction>& rv) const
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
