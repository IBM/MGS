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

#include "InterfaceToMember.h"
#include "InterfaceMapping.h"
#include "Interface.h"
#include "DataType.h"
#include "StructType.h"
#include "GeneralException.h"
#include "InternalException.h"
#include "NotFoundException.h"
#include "DuplicateException.h"
#include "MemberContainer.h"
#include "ArrayType.h"
#include "Constants.h"
#include <memory>
#include <string>
#include <map>
#include <sstream>

InterfaceToMember::InterfaceToMember(Interface* interface) 
   : InterfaceMapping(interface)
{
}

void InterfaceToMember::duplicate(std::auto_ptr<InterfaceToMember>& rv) const
{
   rv.reset(new InterfaceToMember(*this));
}

void InterfaceToMember::duplicate(std::auto_ptr<InterfaceMapping>& rv) const
{
   rv.reset(new InterfaceToMember(*this));
}

void InterfaceToMember::checkAndExtraWork(const std::string& name,
   DataType* member, const DataType* interface, bool amp) {
   
   const DataType* realMember;
      
   const std::vector<std::string>& subAttributePath = 
      member->getSubAttributePath();
   
   if (subAttributePath.size() > 0) {    
      if (member->isArray()) {
	 ArrayType* array = static_cast<ArrayType*>(member);
	 realMember = array->getType();
	 _mappingType.push_back(ONETOMANY);
      } else {
	 realMember = member;
	 _mappingType.push_back(ONETOONE);
      }

      const StructType* nextStruct;
      const DataType* nextMember;
      nextStruct = dynamic_cast<const StructType*>(realMember);

      if (nextStruct == 0) {
	 std::ostringstream os;
 	 os << realMember->getName()
 	    << " is not a struct type"; 
 	 throw GeneralException(os.str());
      }

      if (nextStruct->isPointer()) {
	 std::ostringstream os;
 	 os << nextStruct->getName()
 	    << " can not be a pointer"; 
 	 throw GeneralException(os.str());
      }
      
      std::vector<std::string>::const_iterator it, next, end 
	 = subAttributePath.end();
      for (it = subAttributePath.begin(); it != end; ++it) { 
	 try { 
	    nextMember = 
	       nextStruct->_members.getMember(*it);
	 } catch (NotFoundException& e) {
	    std::ostringstream os;
	    os << *it << " does not exist in struct " 
	       << nextStruct->getTypeName(); 
	    throw GeneralException(os.str());
	 }
	 
	 next = it + 1;
	 if (next == end) {
	    if (nextMember->getTypeString() != interface->getTypeString()) {
	       std::ostringstream os;
	       os << " interface " << _interface->getName() << "'s member " 
		  <<  name 
		  << " is of type " << interface->getTypeString() << " not " 
		  << nextMember->getTypeString() << " ( " 
		  << nextMember->getName() << "'s type)";
	       throw GeneralException(os.str());
	    }    
	 } else {
	    nextStruct = dynamic_cast<const StructType*>(nextMember);
	    
	    if (nextStruct == 0) {
	       std::ostringstream os;
	       os << nextMember->getName()
		  << " is not a struct type"; 
	       throw GeneralException(os.str());
	    }	    

	    if (nextStruct->isPointer()) {
	       std::ostringstream os;
	       os << nextStruct->getName()
		  << " can not be a pointer"; 
	       throw GeneralException(os.str());
	    }
	 }
      }
   } else {
      if (member->isArray() && !interface->isArray()) {
	 ArrayType* array = static_cast<ArrayType*>(member);
	 realMember = array->getType();
	 _mappingType.push_back(ONETOMANY);
      } else {
	 realMember = member;
	 _mappingType.push_back(ONETOONE);
      }

      if (realMember->getDescriptor() != interface->getDescriptor()) {
	 std::ostringstream os;
	 os << " interface " << _interface->getName() << "'s member " <<  name 
	    << " is of type " << interface->getDescriptor() << " not " 
	    << realMember->getDescriptor();
	 throw GeneralException(os.str());
      }
   }
}

InterfaceToMember::~InterfaceToMember() 
{
}

std::string InterfaceToMember::getInterfaceToMemberCode(
   const std::string& tab, std::set<std::string>& requiredIncreases) const
{
   return getInterfaceToMemberCode(tab, requiredIncreases, MachineType::CPU);
}
std::string InterfaceToMember::getInterfaceToMemberCode(
   const std::string& tab, std::set<std::string>& requiredIncreases,
   MachineType mach_type) const
{
   std::string interfaceName = PREFIX + _interface->getName() + "Ptr";   
   std::string memberName;

   std::ostringstream os;
   std::vector<MappingType>::const_iterator mIt = _mappingType.begin();
   const_iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it, ++mIt) {
      const std::vector<std::string>& subAttributePath = 
	 it->getDataType()->getSubAttributePath();
      std::string path = "";
      
      std::vector<std::string>::const_iterator sit, send 
	 = subAttributePath.end();
      for (sit = subAttributePath.begin(); sit != send; ++sit) { 
	 path += "." + *sit;
      }

      memberName = "";
      if (it->getDataType()->isShared()) {
	 memberName = "getNonConstSharedMembers().";
      }
      memberName += it->getDataType()->getName(mach_type);
      std::string getMethod;
      getMethod = interfaceName + "->" + PREFIX + "get_" + 
	 _interface->getName() + "_" + it->getName() + "()";
            
      if (*mIt == ONETOONE) {
	 os << tab << memberName << path << " = " << getMethod << ";\n";
      } else { // ONETOMANY
	 if (path == "") {
	    if (mach_type == MachineType::GPU)
	    {
	       if (it->getDataType()->isArray())
	       {
		  std::string tmpVarName = PREFIX_MEMBERNAME + it->getDataType()->getName() + "_index"; 
		  const DataType* dt_ptr = it->getDataType();
		  os << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n"
		     //<< TAB << memberName << ".insert(" 
		     //<< TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "[" << REF_INDEX << "].insert("
		     << TAB << dt_ptr->getNameRaw(MachineType::GPU) << ".insert("
		     << getMethod << ");\n";
		  os << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n";
		  os << TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_num_elements["
		     << REF_INDEX << "] +=1;\n"
		     << TAB << "int " << tmpVarName << " = " 
		     << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_offset["  
		     << REF_INDEX << "] + " << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME 
		     << dt_ptr->getName() << "_num_elements[" << REF_INDEX << "]-1;\n"
		     << TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() 
		     << ".replace(" << tmpVarName << ", " 
		     << getMethod << ");\n";
		     //<< TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() 
		     //<< "[" << tmpVarName << "] = " 
		     //<< getMethod << ");\n";
		  os << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b\n"
		     << TAB <<REF_CC_OBJECT <<  "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_num_elements[" 
		     << REF_INDEX << "] +=1;\n"
		     << TAB << "int " << tmpVarName << " = " << REF_INDEX << " * " << REF_CC_OBJECT << "->" 
		     << PREFIX_MEMBERNAME << dt_ptr->getName() << "_max_elements + " 
		     << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_num_elements["
		     << REF_INDEX << "]-1;\n"
		     << TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << ".replace(" 
		     << tmpVarName << ", " 
		     << getMethod << ");\n";
		     //<< TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "[" 
		     //<< tmpVarName << "] = " 
		     //<< getMethod << ");\n";
		  os << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5\n"
		     << TAB << "assert(0);\n"
		     << "#endif\n";
	       }
	       else{
		  os << tab << memberName << ".insert(" << getMethod << ");\n";
		  //os << tab << REF_CC_OBJECT << it->getDataType()->getName(mach_type) << ".insert(" << getMethod << ");\n";
	       }
	    }
	    else if (mach_type == MachineType::CPU)
	    {
	       os << tab << memberName << ".insert(" << getMethod << ");\n";
	       // 	    std::string sizeName = PREFIX + memberName + "Size";
	       // 	    os << tab << "int " << sizeName << " = " << memberName 
	       // 	       << ".size();\n"
	       // 	       << tab <<  memberName << ".increase();\n" 
	       // 	       << tab << memberName << "[" << sizeName << "]" 
	       // 	       << " = " << getMethod << ";\n";
	    }
	    else{
	       assert(0);
	    }
	 } else {
	    if (mach_type == MachineType::GPU)
	    {
	       if (it->getDataType()->isArray())
	       {
		  std::string tmpVarName = PREFIX_MEMBERNAME + it->getDataType()->getName() + "_index"; 
		  const DataType* dt_ptr = it->getDataType();
//#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
//         int CG_pyramidalLateralInputsSize = _container->um_pyramidalLateralInputs[index].size();
//         _container->um_pyramidalLateralInputs[index].increase();
//         _container->um_pyramidalLateralInputs[index][CG_pyramidalLateralInputsSize].input = CG_FiringRateProducerPtr->CG_get_FiringRateProducer_output();
//         pyramidalLateralInputs[CG_pyramidalLateralInputsSize].weight = CG_castedPSet->weight;
//#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
//         _container->um_pyramidalLateralInputs_num_elements[index] +=1;
//         int um_pyramidalLateralInputs_index = _container->um_pyramidalLateralInputs_offset[index] + _container->um_pyramidalLateralInputs_num_elements[index]-1;
//         _container->um_pyramidalLateralInputs[um_pyramidalLateralInputs_index].input = CG_FiringRateProducerPtr->CG_get_FiringRateProducer_output();
//         _container->um_pyramidalLateralInputs[um_pyramidalLateralInputs_index].weight = CG_castedPSet->weight;
//#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
//         _container->um_pyramidalLateralInputs_num_elements[index] +=1;
//         int um_pyramidalLateralInputs_index = index * _container->um_pyramidalLateralInputs_max_elements + _container->um_pyramidalLateralInputs_num_elements[index]-1;
//         _container->um_pyramidalLateralInputs[um_pyramidalLateralInputs_index].input = CG_FiringRateProducerPtr->CG_get_FiringRateProducer_output();
//         _container->um_pyramidalLateralInputs[um_pyramidalLateralInputs_index].weight = CG_castedPSet->weight;
//#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
//   assert(0);
//#else
//#endif
//         int CG_pyramidalLateralInputsSize = pyramidalLateralInputs.size();
//         pyramidalLateralInputs.increase();
//         pyramidalLateralInputs[CG_pyramidalLateralInputsSize].input = CG_FiringRateProducerPtr->CG_get_FiringRateProducer_output();
//         pyramidalLateralInputs[CG_pyramidalLateralInputsSize].weight = CG_castedPSet->weight;
		  std::string sizeName = "CG_" + dt_ptr->getName(MachineType::CPU) + "Size"; 
		  std::string tab = TAB + TAB + TAB;
		  os << tab << "{\n";
		  os << tab << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n"
		     << tab << TAB << "int " << sizeName <<  " =" << dt_ptr->getNameRaw(MachineType::GPU) << ".size();\n"
		     << tab << TAB << dt_ptr->getNameRaw(MachineType::GPU) << ".increase();\n"
		     << tab << TAB << dt_ptr->getNameRaw(MachineType::GPU) << "[" << sizeName << "]" << path << " = " << getMethod << ";\n";
		  os << tab << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n";
		  os << tab << TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_num_elements[" << REF_INDEX << "] +=1;\n"
		     << tab << TAB << "int " << tmpVarName << " = " << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_offset[" << REF_INDEX << "] + " << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME 
		     << dt_ptr->getName() << "_num_elements[" << REF_INDEX << "]-1;\n"
		     << tab << TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "[" << tmpVarName << "]" << path << " = " << getMethod << ";\n";
		  os << tab << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b\n"
		     << tab << TAB <<REF_CC_OBJECT <<  "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_num_elements[" << REF_INDEX << "] +=1;\n"
		     << tab << TAB << "int " << tmpVarName << " = " << REF_INDEX << " * " << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_max_elements + " << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_num_elements[" << REF_INDEX << "]-1;\n"
		     << tab << TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "[" << tmpVarName << "]" << path << " = " << getMethod << ";\n";
		  os << tab << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5\n"
		     << tab << TAB << "assert(0);\n"
		     << tab << "#endif\n";
		  os << tab << "}\n";
	       }
	       else{
		  os << tab << memberName << ".insert(" << getMethod << ");\n";
		  //os << tab << REF_CC_OBJECT << it->getDataType()->getName(mach_type) << ".insert(" << getMethod << ");\n";
	       }
	    }
	    else if (mach_type == MachineType::CPU)
	    {
	       std::string sizeName = PREFIX + memberName + "Size";
	       requiredIncreases.insert(memberName);
	       //  	    os << tab << "int " << sizeName << " = " << memberName 
	       //  	       << ".size();\n"
	       // 	       << tab <<  memberName << ".increase();\n";
	       os << tab << memberName << "[" << sizeName << "]" 
		  << path << " = " << getMethod << ";\n";	    
	    }
	 }
      }
   }
   return os.str();
}

std::string InterfaceToMember::getInterfaceToMemberString(
   const std::string& interfaceName) const 
{
   return commonGenerateString(interfaceName, " >> ", "\t\t");
}
