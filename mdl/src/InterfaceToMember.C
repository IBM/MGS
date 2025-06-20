// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

void InterfaceToMember::duplicate(std::unique_ptr<InterfaceToMember>&& rv) const
{
   rv.reset(new InterfaceToMember(*this));
}

void InterfaceToMember::duplicate(std::unique_ptr<InterfaceMapping>&& rv) const
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
   MachineType mach_type,
   bool dummy,
   const std::string& className) const
{
   std::string interfaceName = PREFIX + _interface->getName() + "Ptr";   
   std::string memberName;

   std::ostringstream os;
   std::vector<MappingType>::const_iterator mIt = _mappingType.begin();
   const_iterator it, end = _mappings.end();

   if (dummy)
   {
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
	    /* comment out in dummy setting */
	    os << tab << "//" << memberName << path << " = " << getMethod << ";\n";
	 } else { // ONETOMANY
	    if (path == "") {
	       if (mach_type == MachineType::GPU)
	       {
		  if (it->getDataType()->isArray())
		  {
		     std::string tmpVarName = PREFIX_MEMBERNAME + it->getDataType()->getName(); 
		     os	<< tab << TAB << "if (! sizeIncreased_" << tmpVarName  << ") \n"
			<< tab << TAB << "{\n"
			<< tab << TAB << TAB << 
               "int gridLayerIndex = nd_for_this_node->getGridLayerData()->getGridLayerIndex();\n" 
			<< tab << TAB << TAB << 
               "int nodeAccessor_index = ((NodeInstanceAccessor*)nd_for_this_node)->getIndex();\n" 
			<< tab << TAB << TAB << 
               "std::pair<int, int> pair_data = std::make_pair(gridLayerIndex, nodeAccessor_index);\n"
			<< tab << TAB << TAB << 
               "if (sim->_nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"].count(pair_data) == 0)\n"
			<< tab << TAB << TAB << "{\n"
			<< tab << TAB << TAB << TAB << 
                   "sim->_nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"][pair_data] = 1;\n"
			<< tab << TAB << TAB << "}\n"
			<< tab << TAB << TAB << 
	       "else{\n"
			<< tab << TAB << TAB << TAB << 
                   "sim->_nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"][pair_data] += 1;\n"
			<< tab << TAB << TAB << "}\n"
			<< tab << TAB << TAB << "sizeIncreased_" << tmpVarName << " = true;\n"
			<< tab << TAB << "}\n";
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
		     std::string tmpVarName = PREFIX_MEMBERNAME + it->getDataType()->getName(); 
		     std::string tab = TAB + TAB + TAB;
		     os << tab << "{\n";
		     //os << tab << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n"
		     os	<< tab << TAB << "if (! sizeIncreased_" << tmpVarName  << ") \n"
			<< tab << TAB << "{\n"
			<< tab << TAB << TAB << 
               "int gridLayerIndex = nd_for_this_node->getGridLayerData()->getGridLayerIndex();\n" 
			<< tab << TAB << TAB << 
               "int nodeAccessor_index = ((NodeInstanceAccessor*)nd_for_this_node)->getIndex();\n" 
			<< tab << TAB << TAB << 
               "std::pair<int, int> pair_data = std::make_pair(gridLayerIndex, nodeAccessor_index);\n"
			<< tab << TAB << TAB << 
               "if (sim->_nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"].count(pair_data) == 0)\n"
			<< tab << TAB << TAB << "{\n"
			<< tab << TAB << TAB << TAB << 
                   "sim->_nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"][pair_data] = 1;\n"
			<< tab << TAB << TAB << "}\n"
			<< tab << TAB << TAB << 
	       "else{\n"
			<< tab << TAB << TAB << TAB << 
                   "sim->_nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"][pair_data] += 1;\n"
			<< tab << TAB << TAB << "}\n"
			<< tab << TAB << TAB << "sizeIncreased_" << tmpVarName << " = true;\n"
			<< tab << TAB << "}\n";
			/*
               //sim->_nodes_subarray["LeakyIAFUnit"]["um_inputs"][index] += 1;

               int gridLayerIndex = nd_for_this_node->getGridLayerData()->getGridLayerIndex();
               int nodeAccessor_index = ((NodeInstanceAccessor*)nd_for_this_node)->getIndex();
               std::pair<int, int> pair_data = std::make_pair(gridLayerIndex, nodeAccessor_index); 
               if (sim->_nodes_subarray["LeakyIAFUnit"]["um_inputs"].count(pair_data) == 0)
               {
                   sim->_nodes_subarray["LeakyIAFUnit"]["um_inputs"][pair_data] = 1;
                   //std::map<std::string, size_t> value;
                   //sim->_nodes_subarray_current_index_count["LeakyIAFUnit"]["um_inputs"] = 0;
               }
               else
                   sim->_nodes_subarray["LeakyIAFUnit"]["um_inputs"][pair_data] += 1;

               //_container->um_inputs[index].increase();
               //sizeIncreased = true;
               sizeIncreased_um_inputs = true;
			 */
			//<< tab << TAB << "else{\n"
			//<< tab << TAB << TAB << sizeName << "--;\n"
			//<< tab << TAB << "}\n"
			//<< tab << TAB << dt_ptr->getNameRaw(MachineType::GPU) << ".increase();\n"
			//<< tab << TAB << dt_ptr->getNameRaw(MachineType::GPU) << "[" << sizeName << "]" << path << " = " << getMethod << ";\n";
		//     os << tab << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n"
		//	<< tab << TAB << "if (! sizeIncreased_" << tmpVarName  << ") \n"
		//	<< tab << TAB << "{\n"
		//	<< tab << TAB << TAB << 
		//	    REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_num_elements[" << REF_INDEX << "] +=1;\n"
		//	<< tab << TAB << TAB << 
		//	   "int " << tmpVarName << " = " << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_offset[" << REF_INDEX << "] + " << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_num_elements[" << REF_INDEX << "]-1;\n"
		//	<< tab << TAB << TAB << 
		//	   REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "[" << tmpVarName << "]" << path << " = " << getMethod << ";\n"
		//	<< tab << TAB << "}\n";
		//     os << tab << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b\n"
		//	<< tab << TAB << "if (! sizeIncreased_" << tmpVarName  << ") \n"
		//	<< tab << TAB << "{\n"
		//	<< tab << TAB << TAB << 
		//	   REF_CC_OBJECT <<  "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_num_elements[" << REF_INDEX << "] +=1;\n"
		//	<< tab << TAB << TAB << 
		//	   "int " << tmpVarName << " = " << REF_INDEX << " * " << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_max_elements + " << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_num_elements[" << REF_INDEX << "]-1;\n"
		//	<< tab << TAB << TAB << 
		//	   REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "[" << tmpVarName << "]" << path << " = " << getMethod << ";\n"
		//	<< tab << TAB << "}\n";
		//     os << tab << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5\n"
		//	<< tab << TAB << "assert(0);\n"
		//	<< tab << "#endif\n";
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
   }else{
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
		     //         if (! sizedIncreased)
		     //         {  	sizedIncreased = true; 
		     //        	 _container->um_pyramidalLateralInputs[index].increase();
		     //         }
		     //         else { CG_pyramidalLateralInputsSize -= 1; }
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
			<< tab << TAB << "if (! sizeIncreased) \n"
			<< tab << TAB << "{\n"
			<< tab << TAB << TAB << dt_ptr->getNameRaw(MachineType::GPU) << ".increase();\n"
			<< tab << TAB << TAB << "sizeIncreased = true;\n"
			<< tab << TAB << "}\n"
			<< tab << TAB << "else{\n"
			<< tab << TAB << TAB << sizeName << "--;\n"
			<< tab << TAB << "}\n"
			//<< tab << TAB << dt_ptr->getNameRaw(MachineType::GPU) << ".increase();\n"
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
   }
   return os.str();
}

std::string InterfaceToMember::getInterfaceToMemberString(
   const std::string& interfaceName) const 
{
   return commonGenerateString(interfaceName, " >> ", "\t\t");
}
