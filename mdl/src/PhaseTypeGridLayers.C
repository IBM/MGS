// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "PhaseTypeGridLayers.h"
#include "PhaseType.h"
#include "Class.h"
#include "Method.h"
#include "Constants.h"
#include <memory>
#include <string>
#include <sstream>

PhaseTypeGridLayers::PhaseTypeGridLayers()
   : PhaseType()
{
}

void PhaseTypeGridLayers::duplicate(std::unique_ptr<PhaseType>&& rv) const
{
   rv.reset(new PhaseTypeGridLayers(*this));
}

PhaseTypeGridLayers::~PhaseTypeGridLayers()
{
}

std::string PhaseTypeGridLayers::getType() const
{
   return "GridLayers";
}

std::string PhaseTypeGridLayers::getParameter(
   const std::string& componentType) const
{
   return "GridLayerData* arg";
}

void PhaseTypeGridLayers::generateInstancePhaseMethod( 
   Class& c, const std::string& name, const std::string& instanceType, 
   const std::string& componentType,
   const std::string& workUnitName) const
{
   std::ostringstream os;
   std::unique_ptr<Method> method;
   getInternalInstancePhaseMethod(std::move(method), name, componentType, workUnitName);

   std::string insBaseType = PREFIX + instanceType;

   // Change the cast to static_cast later
   os << TAB << instanceType << "* it = dynamic_cast<" 
      << insBaseType << "GridLayerData*>(arg)->getNodes();\n"
      << TAB << instanceType << "* end = it + arg->getNbrUnits();\n"
      << TAB << "for (; it < end; ++it) {\n"
      << TAB << TAB << "it->" << name << "();\n"
      << TAB << "}\n";      

   method->setFunctionBody(os.str());
   c.addMethod(std::move(method));
}

std::string PhaseTypeGridLayers::getWorkUnitsMethodBody(
   const std::string& tab, const std::string& workUnits,
   const std::string& instanceType, const std::string& name, 
   const std::string& componentType) const
{
   std::ostringstream os;
   
   std::string workUnitName = instanceType + "WorkUnitGridLayers"; 

   os << tab << instanceType << "GridLayerData** it = _gridLayerDataArray;\n"
      << tab << instanceType 
      << "GridLayerData** end = it + _gridLayerDataArraySize;\n"
      << tab << "for (; it < end; ++it) {\n"
      << tab << TAB << "WorkUnit* workUnit = new " << workUnitName 
      << "(*it, &" << instanceType << COMPCATEGORY << "::" 
      << getInstancePhaseMethodName(name, workUnitName) << ", this);\n" 
      //<< tab << TAB << workUnits << ".push_back(workUnit);\n"
      << tab << TAB << "_" << workUnits << "[\"" << name 
      << "\"].push_back(workUnit);\n"
      << tab << "}\n"
      << tab << "_sim.addWorkUnits(getSimulationPhaseName(\"" << name 
      << "\"), _" << workUnits 
      << "[\"" << name << "\"]" << ");\n";

   return os.str();
}

