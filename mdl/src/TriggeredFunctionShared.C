// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "TriggeredFunctionShared.h"
#include "Constants.h"

TriggeredFunctionShared::TriggeredFunctionShared(
   const std::string& name, RunType runType)
   : TriggeredFunction(name, runType)
{
}

void TriggeredFunctionShared::duplicate(
   std::auto_ptr<TriggeredFunction>& rv) const
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
