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

void PhaseTypeGridLayers::duplicate(std::auto_ptr<PhaseType>& rv) const
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
   const std::string& componentType) const
{
   std::ostringstream os;
   std::auto_ptr<Method> method;
   getInternalInstancePhaseMethod(method, name, componentType);

   std::string insBaseType = PREFIX + instanceType;

   // Change the cast to static_cast later
   os << TAB << instanceType << "* it = dynamic_cast<" 
      << insBaseType << "GridLayerData*>(arg)->getNodes();\n"
      << TAB << instanceType << "* end = it + arg->getNbrUnits();\n"
      << TAB << "for (; it < end; ++it) {\n"
      << TAB << TAB << "it->" << name << "();\n"
      << TAB << "}\n";      

   method->setFunctionBody(os.str());
   c.addMethod(method);
}

std::string PhaseTypeGridLayers::getWorkUnitsMethodBody(
   const std::string& tab, const std::string& workUnits,
   const std::string& instanceType, const std::string& name, 
   const std::string& componentType) const
{
   std::ostringstream os;

   os << tab << instanceType << "GridLayerData** it = _gridLayerDataArray;\n"
      << tab << instanceType 
      << "GridLayerData** end = it + _gridLayerDataArraySize;\n"
      << tab << "for (; it < end; ++it) {\n"
      << tab << TAB << "WorkUnit* workUnit = new " << instanceType 
      << "WorkUnitGridLayers(*it, &" << instanceType << COMPCATEGORY << "::" 
      << getInstancePhaseMethodName(name) << ", this);\n" 
      //<< tab << TAB << workUnits << ".push_back(workUnit);\n"
      << tab << TAB << "_" << workUnits << "[\"" << name 
      << "\"].push_back(workUnit);\n"
      << tab << "}\n"
      << tab << "_sim.addWorkUnits(getSimulationPhaseName(\"" << name 
      << "\"), _" << workUnits 
      << "[\"" << name << "\"]" << ");\n";

   return os.str();
}

