// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "PhaseTypeShared.h"
#include "PhaseType.h"
#include "Constants.h"
#include <memory>
#include <string>
#include <sstream>

PhaseTypeShared::PhaseTypeShared()
   : PhaseType()
{
}

void PhaseTypeShared::duplicate(std::unique_ptr<PhaseType>&& rv) const
{
   rv.reset(new PhaseTypeShared(*this));
}

PhaseTypeShared::~PhaseTypeShared()
{
}

std::string PhaseTypeShared::getType() const
{
   return "Shared";
}

std::string PhaseTypeShared::getParameter(
   const std::string& componentType) const
{
   return "";
}

void PhaseTypeShared::generateInstancePhaseMethod( 
   Class& c, const std::string& name, const std::string& instanceType, 
   const std::string& componentType,
   const std::string& workUnitName) const
{
   // nothing
}

std::string PhaseTypeShared::getWorkUnitsMethodBody(
   const std::string& tab, const std::string& workUnits,
   const std::string& instanceType, const std::string& name, 
   const std::string& componentType) const
{
   std::ostringstream os;

   os << tab << "WorkUnit* workUnit = new " << PREFIX << instanceType 
      << "WorkUnitShared(&" << instanceType << COMPCATEGORY << "::" 
      << name << ", this);\n" 
      //<< tab << workUnits << "ma.push_back(workUnit);\n"
      << tab << "_" << workUnits << "[\"" << name 
      << "\"].push_back(workUnit);\n"
      << tab << "_sim.addWorkUnits(getSimulationPhaseName(\"" << name 
      << "\"), _" << workUnits 
      << "[\"" << name << "\"]" << ");\n";

//    os << tab << workUnits << ".push_back(new " << PREFIX << instanceType 
//       << "WorkUnitShared(&" << instanceType << COMPCATEGORY "::" 
//       << name << ", this));\n";

   return os.str();
}
