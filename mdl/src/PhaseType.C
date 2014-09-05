// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
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
   const std::string& name) const
{
   return PREFIX + "InstancePhase_" + name;
}

void PhaseType::getInternalInstancePhaseMethod(
   std::auto_ptr<Method>& method, const std::string& name, 
   const std::string& componentType) const
{
   method.reset(new Method(getInstancePhaseMethodName(name), "void"));
   method->setVirtual();
   std::string parameter = getParameter(componentType);
   if (parameter != "") {
      method->addParameter(parameter);
   }
   method->addParameter("RNG& rng");
}
