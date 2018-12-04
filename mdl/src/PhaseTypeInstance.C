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

#include "PhaseTypeInstance.h"
#include "PhaseType.h"
#include "Class.h"
#include "Method.h"
#include "Constants.h"
#include <memory>
#include <string>
#include <iostream>

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
   const std::string& componentType, const std::string& workUnitName) const
{
   {//the same for all class (default is CPU machine-type)
      std::ostringstream os;
      std::auto_ptr<Method> method;
      getInternalInstancePhaseMethod(method, name, componentType, workUnitName);

      std::string insBaseType = PREFIX + instanceType;

      // not OO may change architecture - sgc
      if (componentType == "Node") {
	 // Change the cast to static_cast later
	 os << "#if defined(HAVE_GPU)\n";
	 os << TAB << "ShallowArray_Flat<"<<instanceType<<">::iterator it = _nodes.begin();\n"
	    << TAB << "ShallowArray_Flat<"<<instanceType<<">::iterator end = _nodes.begin();\n"
	    << "#else\n";
	 os << TAB << "ShallowArray<"<<instanceType<<">::iterator it = _nodes.begin();\n"
	    << TAB << "ShallowArray<"<<instanceType<<">::iterator end = _nodes.begin();\n"
	    << "#endif\n"
	    << TAB << "it += arg->startIndex;\n"
	    << TAB << "end += arg->endIndex;\n"       
	    << TAB << "for (; it <= end; ++it) {\n"
	    << TAB << TAB << "(*it)." << name << "(wu->getRNG());\n"
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
	    << TAB << TAB << "elem->" << name << "(wu->getRNG());\n"
	    << TAB << "}\n";
      } else if (componentType == "Edge") {
	 os
	    << TAB << "ShallowArray<" << instanceType << ">::iterator it" 
	    << " = _edgeList.begin() + arg->startIndex;\n"
	    << TAB << "ShallowArray<" << instanceType << ">::iterator end" 
	    << " = _edgeList.begin() + arg->endIndex;\n"
	    << TAB << "for (; it <= end; ++it) {\n"
	    << TAB << TAB << "it->" << name << "(wu->getRNG());\n"
	    << TAB << "}\n";

      }
      method->setFunctionBody(os.str());
      c.addMethod(method);
   }
   {//special treatment, i.e. get phases for different machine type
      std::ostringstream os;
      //Here is for GPU machine-type
      std::auto_ptr<Method> method;
      getInternalInstancePhaseMethod(method, name, componentType, workUnitName, MachineType::GPU);
      std::string gpuKernelName(instanceType + "_kernel_" + name);
      method->setGPUName(gpuKernelName);

      std::string insBaseType = PREFIX + instanceType;

      // not OO may change architecture - sgc
      if (componentType == "Node") {
	 // Change the cast to static_cast later
	 os 
	  << TAB << "//int BLOCKS= _nodes.size();\n"
          << TAB << "//int THREADS_PER_BLOCK = 1;\n"
          << TAB << "int THREADS_PER_BLOCK= 256;\n"
          << TAB << "int BLOCKS= ceil((float)_nodes.size() / THREADS_PER_BLOCK);\n"
          //<< TAB << "//TUAN TODO: consider using stream later\n"
          //<< TAB << "LifeNode_kernel_initialize<<< BLOCKS_LIFENODE, THREADS_PER_BLOCK_LIFENODE >>> (\n"
          //<< TAB << instanceType << "_kernel_" << method << "<<< BLOCKS, THREADS_PER_BLOCK >>> (\n"
          << TAB << gpuKernelName << "<<< BLOCKS, THREADS_PER_BLOCK >>> (\n"
	  << c.getKernelArgsAsCalledFromCPU() 
          << TAB << ");\n"
          << TAB << "gpuErrorCheck( cudaPeekAtLastError() );\n";

	 method->setFunctionBody(os.str());
	 c.addMethodToExternalFile(instanceType + COMPCATEGORY + ".incl", method);
      } else if (componentType == "Variable") {
	 //nothing
      } else if (componentType == "Edge") 
      {
	 //nothing
      }
   }
}

std::string PhaseTypeInstance::getWorkUnitsMethodBody(
   const std::string& tab, const std::string& workUnits,
   const std::string& instanceType, const std::string& name, 
   const std::string& componentType) const
{
   std::ostringstream os;
  
   std::string partitionItem = componentType + "PartitionItem";
   std::string workUnitName = instanceType + "WorkUnitInstance"; 
   os << tab << "switch(_sim.getPhaseMachineType(\"" << name
      << "\") )\n";
   os << tab << "{\n";
   for(const auto& mt : MachineTypeNames) {    
     os << tab << TAB << "case machineType::" << mt.second << " :\n";
     os << tab << TAB << "{\n";
     os << tab << TAB << TAB <<  partitionItem << "* it = _" << mt.second << "partitions;\n"
	<< tab << TAB << TAB << partitionItem << "* end = it + _nbr"
	<< mt.second << "partitions;\n"
	<< tab << TAB << TAB << "for (; it < end; ++it) {\n"
	<< tab << TAB << TAB << TAB << "WorkUnit* workUnit = \n"
	<< tab << TAB << TAB << TAB << TAB << "new " << workUnitName << "(it,\n"
	<< tab << TAB << TAB << TAB << TAB << TAB << "&" << instanceType << COMPCATEGORY << "::" 
	<< getInstancePhaseMethodName(name, workUnitName, mt.first) << ", this);\n" 
	<< tab << TAB << TAB << TAB << "_" << workUnits << "[\"" << name 
	<< "\"].push_back(workUnit);\n"
	<< tab << TAB << TAB << "}\n"
	<< tab << TAB << TAB << "_sim.addWorkUnits(getSimulationPhaseName(\"" << name 
	<< "\"), _" << workUnits 
	<< "[\"" << name << "\"]" << " );\n"
	<< tab << TAB << "}\n"
	<< tab << TAB << "break;\n\n";
   }
   os << tab << TAB << "default : assert(0); break;\n"
      << tab << "}\n";
	
   /*
   os << tab << partitionItem << "* it = _partitions;\n"
      << tab << partitionItem << "* end = it + _nbrPartitions;\n"
      << tab << "for (; it < end; ++it) {\n"
      << tab << TAB << "WorkUnit* workUnit = new " << workUnitName 
      << "(it, &" << instanceType << COMPCATEGORY << "::" 
      << getInstancePhaseMethodName(name, workUnitName) << ", this);\n" 
      // << tab << TAB << workUnits << ".push_back(workUnit);\n"
      << tab << TAB << "_" << workUnits << "[\"" << name 
      << "\"].push_back(workUnit);\n"
      << tab << "}\n"
      << tab << "_sim.addWorkUnits(getSimulationPhaseName(\"" << name 
      << "\"), _" << workUnits 
      << "[\"" << name << "\"]" << " );\n";
   */
   return os.str();
}
