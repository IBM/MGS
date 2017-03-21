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

#include "ComputeTimeType.h"
#include "Class.h"
#include "Method.h"
#include "Constants.h"
#include <memory>
#include <string>

ComputeTimeType::ComputeTimeType()
{
}

ComputeTimeType::~ComputeTimeType()
{
}

std::string ComputeTimeType::getInstanceComputeTimeMethodName(
   const std::string& name) const
{
   return PREFIX + "InstanceComputeTime_" + name;
}

void ComputeTimeType::getInternalInstanceComputeTimeMethod(
   std::auto_ptr<Method>& method, const std::string& name, 
   const std::string& componentType) const
{
   method.reset(new Method(getInstanceComputeTimeMethodName(name), "void"));
   method->setVirtual();
   std::string parameter = getParameter(componentType);
   if (parameter != "") {
      method->addParameter(parameter);
   }
}
