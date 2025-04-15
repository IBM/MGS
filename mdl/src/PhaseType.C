// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "PhaseType.h"
#include "Class.h"
#include "Method.h"
#include "Constants.h"
#include <memory>
#include <string>

PhaseType::PhaseType()
{
}

PhaseType::~PhaseType()
{
}

std::string PhaseType::getInstancePhaseMethodName(
						  const std::string& name,
						  const std::string& workUnitName,
						  MachineType mach_type) const
{
   std::string method_name(PREFIX + "InstancePhase_" + name);
   if (mach_type == MachineType::GPU)
      method_name = PREFIX + "host_" + name;
   return method_name;
}

void PhaseType::getInternalInstancePhaseMethod(
   std::unique_ptr<Method>&& method, const std::string& name, 
   const std::string& componentType, const std::string& workUnitName,
   MachineType mach_type) const
{
   method.reset(new Method(getInstancePhaseMethodName(name, workUnitName, mach_type), "void"));
   method->setVirtual();
   std::string parameter = getParameter(componentType);
   if (parameter != "") {
      method->addParameter(parameter);
   }
   method->addParameter(workUnitName + "* wu");
}
