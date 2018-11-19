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
   const std::string& name, MachineType mach_type) const
{
   std::string method_name(PREFIX + "InstancePhase_" + name);
   if (mach_type == MachineType::GPU)
      method_name = PREFIX + "host_" + name;
   return method_name;
}

void PhaseType::getInternalInstancePhaseMethod(
   std::auto_ptr<Method>& method, const std::string& name, 
   const std::string& componentType, 
   MachineType mach_type) const
{
   method.reset(new Method(getInstancePhaseMethodName(name, mach_type), "void"));
   method->setVirtual();
   std::string parameter = getParameter(componentType);
   if (parameter != "") {
      method->addParameter(parameter);
   }
   method->addParameter("RNG& rng");
}
