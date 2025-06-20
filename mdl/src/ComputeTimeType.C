// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
   std::unique_ptr<Method>&& method, const std::string& name, 
   const std::string& componentType) const
{
   method.reset(new Method(getInstanceComputeTimeMethodName(name), "void"));
   method->setVirtual();
   std::string parameter = getParameter(componentType);
   if (parameter != "") {
      method->addParameter(parameter);
   }
}
