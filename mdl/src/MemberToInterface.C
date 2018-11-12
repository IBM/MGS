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

#include "MemberToInterface.h"
#include "InterfaceMapping.h"
#include "Interface.h"
#include "DataType.h"
#include "StructType.h"
#include "GeneralException.h"
#include "InternalException.h"
#include "NotFoundException.h"
#include "DuplicateException.h"
#include "MemberContainer.h"
#include "Class.h"
#include "Method.h"
#include "Constants.h"
#include <memory>
#include <string>
#include <map>
#include <iostream>
#include <sstream>

MemberToInterface::MemberToInterface(Interface* interface) 
   : InterfaceMapping(interface)
{
}

void MemberToInterface::duplicate(std::auto_ptr<MemberToInterface>& rv) const
{
   rv.reset(new MemberToInterface(*this));
}

void MemberToInterface::duplicate(std::auto_ptr<InterfaceMapping>& rv) const
{
   rv.reset(new MemberToInterface(*this));
}

bool MemberToInterface::checkAllMapped() 
{
   bool retVal = true;
   MemberContainer<DataType>::const_iterator it, 
      end = _interface->getMembers().end();
   for (it = _interface->getMembers().begin(); it != end; it++) {
      if (!existsInMappings(it->first)) {
	 retVal = false;
	 break;	 
      }
   }
   return retVal;
}

void MemberToInterface::checkAndExtraWork(const std::string& name,
   DataType* member, const DataType* interface, bool amp) {
   const std::vector<std::string>& subAttributePath = 
      member->getSubAttributePath();
   
   if (subAttributePath.size() > 0) {
      StructType* nextStruct;
      DataType* nextMember;
      nextStruct = dynamic_cast<StructType*>(member);

      if (nextStruct == 0) {
	 std::ostringstream os;
 	 os << member->getName()
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
	    std::string memberTypeString = nextMember->getTypeString();
	    if (amp) {
	       memberTypeString += "*";
	    }
	    if (memberTypeString != interface->getTypeString()) {
	       std::ostringstream os;
	       os << " interface " << _interface->getName() << "'s member " 
		  <<  name 
		  << " is of type " << interface->getDescriptor() << " not " 
		  << memberTypeString << " ( " 
		  << nextMember->getName() << "'s type)";
	       throw GeneralException(os.str());
	    }    
	 } else {
	    nextStruct = dynamic_cast<StructType*>(nextMember);
	    
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
      // The implemented interface dataType has to be the pointer of this
      // type, we'll serve &x.
      std::string memberTypeString = member->getTypeString();
      if (amp) {
	 memberTypeString += "*";
      }
      std::string interfaceTypeString = interface->getTypeString();
      if (memberTypeString != interfaceTypeString) {
	 std::ostringstream os;
	 os << " interface " << _interface->getName() << "'s member " <<  name 
	    << " is of type " << interfaceTypeString << " not " 
	    << memberTypeString;
	 throw GeneralException(os.str());
      }
   }
}

MemberToInterface::~MemberToInterface() 
{
}
 

std::string MemberToInterface::getMemberToInterfaceString(
   const std::string& interfaceName) const 
{
   return commonGenerateString(interfaceName, " << ", "\t");
}

void MemberToInterface::setupAccessorMethods(Class& instance) const
{
   const_iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
      const std::vector<std::string>& subAttributePath = 
	 it->getDataType()->getSubAttributePath();
      std::string path = "";
      
      std::vector<std::string>::const_iterator sit, send 
	 = subAttributePath.end();
      for (sit = subAttributePath.begin(); sit != send; ++sit) { 
	 path += "." + *sit;
      }

      std::auto_ptr<Method> method(
	 new Method(PREFIX + "get_" + _interface->getName() + "_" + 
		    it->getName(),
		    it->getTypeString()));
      std::string name = it->getDataType()->getName();
      if (it->getDataType()->isShared()) {
	 name = "getNonConstSharedMembers()." + name;
      }
      name += path;
      if (it->getDataType()->isShared()) {
	 method->setFunctionBody(
	       (!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
	       + TAB + "return " 
	       + (it->getNeedsAmpersand() ? "&" : "") 
	       + name + ";\n");
      }else 
      {
	 std::string body;
	 body = STR_GPU_CHECK_START + "\n"
	    +
	    (!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
	    + TAB + "return " 
	    + (it->getNeedsAmpersand() ? "&" : "") 
	    + "("+REF_CC_OBJECT+"->" + PREFIX_MEMBERNAME  
	    + name + "["+REF_INDEX+"]);\n"
	    +
	    "#else\n"
	    +
	    (!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
	    + TAB + "return " 
	    + (it->getNeedsAmpersand() ? "&" : "") 
	    + name + ";\n"
	    + STR_GPU_CHECK_END + "\n" ;
	 method->setFunctionBody(
	       body
	       );
      }
      method->setVirtual();
      instance.addMethod(method);
   }      
}

void MemberToInterface::setupProxyAccessorMethods(Class& instance) const
{
   const_iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
      const std::vector<std::string>& subAttributePath = 
	 it->getDataType()->getSubAttributePath();
      std::string path = "";
      
      std::vector<std::string>::const_iterator sit, send 
	 = subAttributePath.end();
      for (sit = subAttributePath.begin(); sit != send; ++sit) { 
	 path += "." + *sit;
      }

      std::auto_ptr<Method> method(
	 new Method(PREFIX + "get_" + _interface->getName() + "_" + 
		    it->getName(),
		    it->getTypeString()));
      std::string name = it->getDataType()->getName();
      if (it->getDataType()->isShared()) {
	 name = "getNonConstSharedMembers()." + name;
      }
      name += path;
      method->setFunctionBody(
	 TAB + "return &" 
	 + name + ";\n");
      method->setVirtual();
      instance.addMethod(method);
   }      
}

bool MemberToInterface::hasMemberDataType(const std::string& name) const
{
   std::vector<InterfaceMappingElement>::const_iterator it, 
      end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
      const DataType* dt = it->getDataType();
      if (!dt->isShared()) {
	 if (dt->getName() == name) {
	    return true;
	 }
      }
   }
   return false;
}

std::string MemberToInterface::getServiceNameCode(
   const std::string& tab) const
{
   std::ostringstream os;
   os << tab << "if (" << INTERFACENAME << " == \"" 
      << _interface->getName() << "\") {\n";
   
   std::string newTab = tab + TAB;
   std::vector<InterfaceMappingElement>::const_iterator it, 
      end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
      os << it->getServiceNameCode(newTab);
   }
   os << tab << "}\n";
   return os.str();
}
