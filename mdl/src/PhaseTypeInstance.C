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

#include "PhaseTypeInstance.h"
#include "PhaseType.h"
#include "Class.h"
#include "Method.h"
#include "Constants.h"
#include <memory>
#include <string>

PhaseTypeInstance::PhaseTypeInstance()
   : PhaseType()
{
}

void PhaseTypeInstance::duplicate(std::auto_ptr<PhaseType>& rv) const
{
   rv.reset(new PhaseTypeInstance(*this));
}

PhaseTypeInstance::~PhaseTypeInstance()
{
}

std::string PhaseTypeInstance::getType() const
{
   return "Instance";
}

std::string PhaseTypeInstance::getParameter(
   const std::string& componentType) const
{
   return componentType + "PartitionItem* arg";
}

void PhaseTypeInstance::generateInstancePhaseMethod( 
   Class& c, const std::string& name, const std::string& instanceType, 
   const std::string& componentType) const
{
   std::ostringstream os;
   std::auto_ptr<Method> method;
   getInternalInstancePhaseMethod(method, name, componentType);

   std::string insBaseType = PREFIX + instanceType;
   
   // not OO may change architecture - sgc
   if (componentType == "Node") {
      // Change the cast to static_cast later
     os << TAB << "ShallowArray<"<<instanceType<<">::iterator it = _nodes.begin();\n"
	<< TAB << "ShallowArray<"<<instanceType<<">::iterator end = _nodes.begin();\n"
        << TAB << "it += arg->startIndex;\n"
        << TAB << "end += arg->endIndex;\n"       
	<< TAB << "for (; it <= end; ++it) {\n"
	<< TAB << TAB << "(*it)." << name << "(rng);\n"
	<< TAB << "}\n";
   } else if (componentType == "Variable") {
      os
	 << TAB << "DuplicatePointerArray<Variable>::iterator it" 
	 << " = _variableList.begin() + arg->startIndex;\n"
	 << TAB << "DuplicatePointerArray<Variable>::iterator end" 
	 << " = _variableList.begin() + arg->endIndex;\n"
	 << TAB << "for (; it <= end; ++it) {\n"
	 << TAB << TAB << instanceType << "* elem = dynamic_cast<" 
	 << instanceType << "*>(*it);\n"
	 // for now
	 << TAB << TAB << "assert(elem != 0); // for now \n"
	 << TAB << TAB << "elem->" << name << "(rng);\n"
	 << TAB << "}\n";
   } else if (componentType == "Edge") {
      os
	 << TAB << "ShallowArray<" << instanceType << ">::iterator it" 
	 << " = _edgeList.begin() + arg->startIndex;\n"
	 << TAB << "ShallowArray<" << instanceType << ">::iterator end" 
	 << " = _edgeList.begin() + arg->endIndex;\n"
	 << TAB << "for (; it <= end; ++it) {\n"
	 << TAB << TAB << "it->" << name << "(rng);\n"
	 << TAB << "}\n";

   }

   method->setFunctionBody(os.str());
   c.addMethod(method);
}

std::string PhaseTypeInstance::getWorkUnitsMethodBody(
   const std::string& tab, const std::string& workUnits,
   const std::string& instanceType, const std::string& name, 
   const std::string& componentType) const
{
   std::ostringstream os;

   std::string partitionItem = componentType + "PartitionItem";

   os << tab << partitionItem << "* it = _partitions;\n"
      << tab << partitionItem << "* end = it + _nbrPartitions;\n"
      << tab << "for (; it < end; ++it) {\n"
      << tab << TAB << "WorkUnit* workUnit = new " << instanceType 
      << "WorkUnitInstance(it, &" << instanceType << COMPCATEGORY << "::" 
      << getInstancePhaseMethodName(name) << ", this);\n" 
      // << tab << TAB << workUnits << ".push_back(workUnit);\n"
      << tab << TAB << "_" << workUnits << "[\"" << name 
      << "\"].push_back(workUnit);\n"
      << tab << "}\n"
      << tab << "_sim.addWorkUnits(getSimulationPhaseName(\"" << name 
      << "\"), _" << workUnits 
      << "[\"" << name << "\"]" << " );\n";

   return os.str();
}
