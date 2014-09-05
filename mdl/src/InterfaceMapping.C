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

#include "InterfaceMapping.h"
#include "Interface.h"
#include "DataType.h"
#include "GeneralException.h"
#include "InternalException.h"
#include "NotFoundException.h"
#include "DuplicateException.h"
#include "MemberContainer.h"
#include "InterfaceMappingElement.h"
#include <memory>
#include <string>
#include <map>
#include <iostream>

InterfaceMapping::InterfaceMapping(Interface* interface) 
{
   setInterface(interface);
}

void InterfaceMapping::addMapping(const std::string& name, 
				  std::auto_ptr<DataType>& data,
				  bool amp) 
{
   iterator it = find(name);
   if (it != _mappings.end()) {
      std::ostringstream stream;
      stream << name << " is already in the container.";
      throw DuplicateException(stream.str()); 
   } 
   if (_interface == 0) {
      throw InternalException(
	 "_interface is 0 in InterfaceMapping::addMapping");
   }
   const DataType* type;
   try {
      type = _interface->getMembers().getMember(name);
   } catch (NotFoundException& e) {
      std::string mes = "Member " + name + " is not found on interface " 
	 + _interface->getName(); 
      e.setError(mes);
      throw;
   }
   checkAndExtraWork(name, data.get(), type, amp);
   InterfaceMappingElement elem(name, data, type->getTypeString(), amp);
   _mappings.push_back(elem);
}

InterfaceMapping::~InterfaceMapping() 
{
}


bool InterfaceMapping::existsInMappings(const std::string& token) const
{
   const_iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
      if (it->getName() == token) {
	 return true;
      }
   }
   return false;
}

InterfaceMapping::iterator InterfaceMapping::find(
   const std::string& token)
{
   InterfaceMapping::iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
      if (it->getName() == token) {
	 break;
      }
   }
   return it;
}

std::string InterfaceMapping::commonGenerateString(
   const std::string& interfaceName, const std::string& direction,
   const std::string& tab) const
{
   std::ostringstream os;
   const_iterator it, end = _mappings.end();
   for (it = _mappings.begin(); it != end; ++it) {
	 os  << tab << interfaceName << "." << it->getName() << direction;
	 if (it->getNeedsAmpersand()) {
	    os << "&";
	 }
	 if (it->getDataType()->isShared()) {
	    os << "Shared.";
	 }
	 os << it->getDataType()->getName() << ";\n"; 
   }   
   return os.str();
}
