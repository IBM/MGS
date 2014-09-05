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

void PhaseTypeShared::duplicate(std::auto_ptr<PhaseType>& rv) const
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
   const std::string& componentType) const
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
      //<< tab << workUnits << ".push_back(workUnit);\n"
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
