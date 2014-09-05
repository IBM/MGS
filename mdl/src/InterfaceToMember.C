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
      memberName += it->getDataType()->getName();
      std::string getMethod;
      getMethod = interfaceName + "->" + PREFIX + "get_" + 
	 _interface->getName() + "_" + it->getName() + "()";
            
      if (*mIt == ONETOONE) {
	 os << tab << memberName << path << " = " << getMethod << ";\n";
      } else { // ONETOMANY
	 if (path == "") {
	    os << tab << memberName << ".insert(" << getMethod << ");\n";
// 	    std::string sizeName = PREFIX + memberName + "Size";
// 	    os << tab << "int " << sizeName << " = " << memberName 
// 	       << ".size();\n"
// 	       << tab <<  memberName << ".increase();\n" 
// 	       << tab << memberName << "[" << sizeName << "]" 
// 	       << " = " << getMethod << ";\n";
	 } else {
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
   return os.str();
}

std::string InterfaceToMember::getInterfaceToMemberString(
   const std::string& interfaceName) const 
{
   return commonGenerateString(interfaceName, " >> ", "\t\t");
}
