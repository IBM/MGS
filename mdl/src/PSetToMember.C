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

#include "PSetToMember.h"
#include "StructType.h"
#include "DataType.h"
#include "ArrayType.h"
#include "GeneralException.h"
#include "InternalException.h"
#include "NotFoundException.h"
#include "DuplicateException.h"
#include "MemberContainer.h"
#include "Constants.h"
#include <memory>
#include <string>
#include <map>
#include <iostream>

PSetToMember::PSetToMember(StructType* pset) 
{
   setPSet(pset);
}

void PSetToMember::addMapping(const std::string& name, 
			      std::unique_ptr<DataType>&& data) 
{
   iterator it = find(name);
   if (it != _mappings.end()) {
      std::ostringstream stream;
      stream << name << " is already in the container.";
      throw DuplicateException(stream.str()); 
   } 
   if (_pset == 0) {
      throw InternalException(
	 "_pset is 0 in PSetToMember::addMapping");
   }
   DataType* type;
   try {
      type = _pset->_members.getMember(name);
   } catch (NotFoundException& e) {
      std::string mes = "Member " + name + " is not found on PSet " 
	 + _pset->getName(); 
      e.setError(mes);
      throw;
   }
   checkAndExtraWork(name, data.get(), type);
   elemType elem;
   elem.first = name;
   elem.second = data.release();
   _mappings.push_back(elem);
}

PSetToMember::~PSetToMember() 
{
   destructOwnedHeap();
}


bool PSetToMember::existsInMappings(const std::string& token) const
{
   const_iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
      if (it->first == token) {
	 return true;
      }
   }
   return false;
}

PSetToMember::iterator PSetToMember::find(
   const std::string& token)
{
   PSetToMember::iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
      if (it->first == token) {
	 break;
      }
   }
   return it;
}

std::string PSetToMember::getPSetToMemberString() const
{
   std::string direction = " >> ";
   std::string tab = "\t\t";
   std::ostringstream os;
   const_iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
	 os  << tab << "PSet." << it->first << direction;
	 if (it->second->isShared()) {
	    os << "Shared.";
	 }
	 os << it->second->getName() << ";\n"; 
   }   
   return os.str();
}

void PSetToMember::duplicate(std::unique_ptr<PSetToMember>&& rv) const
{
   rv.reset(new PSetToMember(*this));
}

void PSetToMember::checkAndExtraWork(const std::string& name,
   DataType* member, const DataType* pset) {
   
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
	    if (nextMember->getTypeString() != pset->getTypeString()) {
	       std::ostringstream os;
	       os << " ParameterSets member " 
		  <<  name 
		  << " is of type " << pset->getTypeString() << " not " 
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
      if (member->isArray() && !pset->isArray()) {
	 ArrayType* array = static_cast<ArrayType*>(member);
	 realMember = array->getType();
	 _mappingType.push_back(ONETOMANY);
      } else {
	 realMember = member;
	 _mappingType.push_back(ONETOONE);
      }

      if (realMember->getDescriptor() != pset->getDescriptor()) {
	 std::ostringstream os;
	 os << " pset " << _pset->getName() << "'s member " <<  name 
	    << " is of type " << pset->getDescriptor() << " not " 
	    << realMember->getDescriptor();
	 throw GeneralException(os.str());
      }
   }
}

std::string PSetToMember::getPSetToMemberCode(
   const std::string& tab, std::set<std::string>& requiredIncreases) const
{
   return getPSetToMemberCode(tab, requiredIncreases, MachineType::CPU);
}
std::string PSetToMember::getPSetToMemberCode(
   const std::string& tab, std::set<std::string>& requiredIncreases,
   MachineType mach_type,
   bool dummy, const std::string& className) const
{
   std::string psetName = INATTRPSETNAME;   
   std::string memberName;

   std::ostringstream os;
   std::vector<MappingType>::const_iterator mIt = _mappingType.begin();
   const_iterator it, end = _mappings.end();
   if (dummy)
   {
      for (it = _mappings.begin(); it != end; ++it, ++mIt) {
	 const std::vector<std::string>& subAttributePath = 
	    it->second->getSubAttributePath();
	 std::string path = "";

	 std::vector<std::string>::const_iterator sit, send 
	    = subAttributePath.end();
	 for (sit = subAttributePath.begin(); sit != send; ++sit) { 
	    path += "." + *sit;
	 }

	 memberName = "";
	 if (it->second->isShared()) {
	    memberName = "getNonConstSharedMembers().";
	 }
	 memberName += it->second->getName();
	 if (*mIt == ONETOONE) {
	    os << tab << memberName << " = " << psetName << "->" 
	       << PREFIX << "get_" << _pset->getName() << "_" << it->first 
	       << "();\n";
	 } else { // ONETOMANY
	    std::string getMethod;
	    getMethod = psetName + "->" + it->first;
	    if (path == "") {
	       if (mach_type == MachineType::GPU)
	       {
		  if (it->second->isArray())
		  {
		     std::string tmpVarName = PREFIX_MEMBERNAME + it->second->getName(); 
		     std::string tab = TAB + TAB + TAB;
		     os << tab << "{\n";
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
		     os << tab << "}\n";
		  }
		  else{
		     os << tab << memberName << ".insert(" << psetName << "->" 
			<< it->first << ");\n";
		  }
	       }
	       else if (mach_type == MachineType::CPU)
	       {
		  os << tab << memberName << ".insert(" << psetName << "->" 
		     << it->first << ");\n";
	       }
	       else{
		  assert(0);
	       }
	    } else {
	       //pyramidalLateralInputs[CG_pyramidalLateralInputsSize].weight = CG_castedPSet->weight;
	       if (mach_type == MachineType::GPU)
	       {
		  if (it->second->isArray())
		  {
		     std::string tmpVarName = PREFIX_MEMBERNAME + it->second->getName();
		     std::string tab = TAB + TAB + TAB;
		     os << tab << "{\n";
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
		  os << tab << memberName << "[" << sizeName << "]" 
		     << path << " = " << psetName << "->" 
		     << it->first << ";\n";	    
	       }
	    }
	 }
      }
   }else{
      for (it = _mappings.begin(); it != end; ++it, ++mIt) {
	 const std::vector<std::string>& subAttributePath = 
	    it->second->getSubAttributePath();
	 std::string path = "";

	 std::vector<std::string>::const_iterator sit, send 
	    = subAttributePath.end();
	 for (sit = subAttributePath.begin(); sit != send; ++sit) { 
	    path += "." + *sit;
	 }

	 memberName = "";
	 if (it->second->isShared()) {
	    memberName = "getNonConstSharedMembers().";
	 }
	 memberName += it->second->getName();
	 if (*mIt == ONETOONE) {
	    os << tab << memberName << " = " << psetName << "->" 
	       << PREFIX << "get_" << _pset->getName() << "_" << it->first 
	       << "();\n";
	 } else { // ONETOMANY
	    std::string getMethod;
	    getMethod = psetName + "->" + it->first;
	    if (path == "") {
	       if (mach_type == MachineType::GPU)
	       {
		  if (it->second->isArray())
		  {
		     std::string tmpVarName = PREFIX_MEMBERNAME + it->second->getName() + "_index"; 
		     const DataType* dt_ptr = it->second;
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
		     os << tab << memberName << ".insert(" << psetName << "->" 
			<< it->first << ");\n";
		  }
	       }
	       else if (mach_type == MachineType::CPU)
	       {
		  os << tab << memberName << ".insert(" << psetName << "->" 
		     << it->first << ");\n";
	       }
	       else{
		  assert(0);
	       }
	    } else {
	       //pyramidalLateralInputs[CG_pyramidalLateralInputsSize].weight = CG_castedPSet->weight;
	       if (mach_type == MachineType::GPU)
	       {
		  if (it->second->isArray())
		  {
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
		     std::string tmpVarName = PREFIX_MEMBERNAME + it->second->getName() + "_index"; 
		     const DataType* dt_ptr = it->second;
		     std::string sizeName = "CG_" + dt_ptr->getName(MachineType::CPU) + "Size"; 
		     std::string tab = TAB + TAB + TAB;
		     os << tab << "{\n";
		     os << tab << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n"
			<< tab << TAB << "int " << sizeName <<  " =" << dt_ptr->getNameRaw(MachineType::GPU) << ".size();\n"
			//<< tab << TAB << dt_ptr->getNameRaw(MachineType::GPU) << ".increase();\n"

			<< tab << TAB << "if (! sizeIncreased) \n"
			<< tab << TAB << "{\n"
			<< tab << TAB << TAB << dt_ptr->getNameRaw(MachineType::GPU) << ".increase();\n"
			<< tab << TAB << TAB << "sizeIncreased = true;\n"
			<< tab << TAB << "}\n"
			<< tab << TAB << "else{\n"
			<< tab << TAB << TAB << sizeName << "--;\n"
			<< tab << TAB << "}\n"

			<< tab << TAB << dt_ptr->getNameRaw(MachineType::GPU) << "[" << sizeName << "]" << path << " = " << getMethod << ";\n";
		     os << tab << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n";
		     os << tab << TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_num_elements[" << REF_INDEX << "] +=1;\n"
			<< tab << TAB << "int " << tmpVarName << " = " << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_offset["  << REF_INDEX << "] + " << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << dt_ptr->getName() << "_num_elements[" << REF_INDEX << "]-1;\n"
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
		  os << tab << memberName << "[" << sizeName << "]" 
		     << path << " = " << psetName << "->" 
		     << it->first << ";\n";	    
	       }
	    }
	 }
      }
   }
   return os.str();
}

PSetToMember::PSetToMember(const PSetToMember& rv)
   : _pset(rv._pset), _mappingType(rv._mappingType)
{
   copyOwnedHeap(rv);
}

PSetToMember& PSetToMember::operator=(const PSetToMember& rv)
{
   if (this != &rv) {
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _pset = rv._pset; 
      _mappingType = rv._mappingType;
   }
   return *this;
}

void PSetToMember::destructOwnedHeap()
{
   std::vector<elemType>::iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
      delete it->second;
   }
   _mappings.clear();
}

void PSetToMember::copyOwnedHeap(const PSetToMember& rv)
{
   if (rv._mappings.size() > 0) {
      std::vector<elemType>::const_iterator it, end = rv._mappings.end();
      for (it = _mappings.begin(); it != end; ++it) {
	 elemType elem;
	 elem.first = it->first;
	 std::unique_ptr<DataType> dup;
	 it->second->duplicate(std::move(dup));
	 elem.second = dup.release();
	 _mappings.push_back(elem);
      }      
   }
}
