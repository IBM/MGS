// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Connection.h"
#include "MemberContainer.h"
#include "InterfaceToMember.h"
#include "InternalException.h"
#include "Predicate.h"
#include "StructType.h"
#include "Interface.h"
#include "InterfaceToMember.h"
#include "Constants.h"
#include "ConnectionCCBase.h"
#include "NotFoundException.h"
#include "GeneralException.h"
#include "ConnectionException.h"
#include <memory>
#include <string>
#include <set>
#include <sstream>
#include <cassert>
#include <iostream>

Connection::Connection(DirectionType directionType,
		       ComponentType componentType, 
		       bool graph) 
   : _directionType(directionType), _componentType(componentType),
     _graph(graph)
{
}

bool Connection::getGraph() const
{
   return _graph;
}

void Connection::setGraph(bool graph) 
{
   _graph = graph;
}

std::string Connection::getString() const
{
   std::ostringstream os;
   
   os << "\t" << getTypeString() << " ("
      << getPredicateString()
      << ") Expects ";
   if (_interfaces.size() != 0) {
      MemberContainer<InterfaceToMember>::const_iterator it, next, end 
	 = _interfaces.end();
      for (it = _interfaces.begin(); it != end; it++) {
	 os << it->first;
	 next = it;
	 next++;
	 if ( next != end) {
	    os << ",";
	 }
	 os << " ";
      }
      os << "{\n";
   } else {
      throw InternalException(
	 "_interfaces.size() is 0 in Connection::getString");
   }
   MemberContainer<InterfaceToMember>::const_iterator it, 
      end = _interfaces.end();
   for (it = _interfaces.begin(); it != end; it ++) {
      os << it->second->getInterfaceToMemberString(it->first);
   }
   os << _psetMappings.getPSetToMemberString();
   os << getUserFunctionCallsString()
      << "\t}\n";
   return os.str();
}

Connection::~Connection() 
{
}

std::set<std::string> Connection::getInterfaceCasts(
   const std::string& componentName) const
{
   std::set<std::string> casts;
   MemberContainer<InterfaceToMember>::const_iterator it, 
      end = _interfaces.end();
   for (it = _interfaces.begin(); it != end; ++it) {
      std::string interfaceName = it->first;
      std::string interfaceCast = 
	 interfaceName + "* " 
	 + PREFIX + interfaceName + "Ptr = dynamic_cast<" + interfaceName 
	 + "*>(" + componentName;
      if (_componentType==Connection::_NODE) interfaceCast += "->getNode()";
      interfaceCast += ");\n";
      casts.insert(interfaceCast);
   }
   return casts;
}

std::set<std::string> Connection::getInterfaceNames() const
{
   std::set<std::string> names;
   MemberContainer<InterfaceToMember>::const_iterator it, 
      end = _interfaces.end();
   for (it = _interfaces.begin(); it != end; ++it) {
      names.insert(it->first);
   }
   return names;
}

std::string Connection::getCommonConnectionCode(const std::string& tab,
						const std::string& name) const
{
   std::ostringstream os;

   std::set<std::string> interfaceNames = getInterfaceNames();
   std::set<std::string>::iterator it, end = interfaceNames.end();
   for (it = interfaceNames.begin(); it != end; ++it) {
      os << tab << "if (" << PREFIX << *it << "Ptr == 0) {\n"
	 << tab << TAB << "std::cerr << \"Dynamic Cast of "
	 << *it << " failed in " << name
	 << "\" << std::endl;\n"
	 << tab << TAB << "exit(-1);\n"
	 << tab << "}\n"
	 << "\n";
   }

   MemberContainer<InterfaceToMember>::const_iterator it2, 
      end2 = _interfaces.end();
   
   std::set<std::string> requiredIncreases;
   std::string interfaceToMemberCodes;
   std::string psetToMemberCodes;
   for (it2 = _interfaces.begin(); it2 != end2; ++it2) {
      interfaceToMemberCodes += 
	 it2->second->getInterfaceToMemberCode(tab, requiredIncreases);
   }
   psetToMemberCodes = _psetMappings.getPSetToMemberCode(tab, 
							 requiredIncreases);

   end = requiredIncreases.end();
   for (it = requiredIncreases.begin(); it != end; ++it) {
      std::string sizeName = PREFIX + *it + "Size";
      os << tab << "int " << sizeName << " = " << *it 
	 << ".size();\n"
	 << tab << *it << ".increase();\n";
   }
   

   os << interfaceToMemberCodes
      << psetToMemberCodes;
   return os.str();   
}

std::string Connection::getCommonConnectionCodeAlternativeInterfaceSet(const std::string& tab,
						const std::string& name, const std::string& predicate) const
{
   return getCommonConnectionCodeAlternativeInterfaceSet(tab, name, predicate, MachineType::CPU); 
}
std::string Connection::getCommonConnectionCodeAlternativeInterfaceSet(const std::string& tab,
						const std::string& name, const std::string& predicate,
						MachineType mach_type,
						bool dummy) const
{
   std::ostringstream os;

   if (dummy)
   {
      // code in addPreNode_Dummy(..)
      
      std::set<std::string> interfaceNames = getInterfaceNames();
      std::set<std::string>::iterator it, end = interfaceNames.end();
      os << tab << "bool castMatchLocal = true;\n";
      os << tab << "noPredicateMatch = true;\n";
      //for (it = interfaceNames.begin(); it != end; ++it) {
      //   os << tab << "if (" << PREFIX << *it << "Ptr == 0) {\n"
      //      << "#if !defined(NOWARNING_DYNAMICCAST) \n"
      //      << tab << TAB << "std::cerr << \"Dynamic Cast of "
      //      << *it << " failed in " << name
      //      << "\" << std::endl;\n"
      //      << "#endif\n"
      //      //<< tab << TAB << "exit(-1);\n"
      //      << tab << TAB << "castMatchLocal = false;\n"
      //      << tab << "}\n"
      //      << "\n";
      //}

      MemberContainer<InterfaceToMember>::const_iterator it2, 
	 end2 = _interfaces.end();

      std::set<std::string> requiredIncreases;
      std::string interfaceToMemberCodes;
      std::string psetToMemberCodes;
      std::string subtab = tab + TAB;
      for (it2 = _interfaces.begin(); it2 != end2; ++it2) {
	 interfaceToMemberCodes += 
	    it2->second->getInterfaceToMemberCode(subtab, requiredIncreases, mach_type, dummy, name);
      }
      psetToMemberCodes = _psetMappings.getPSetToMemberCode(subtab, 
	    requiredIncreases, mach_type, dummy, name);

      end = requiredIncreases.end();
      os << tab <<  "if (castMatchLocal) { \n";
      os << tab << TAB <<  "if (matchPredicateAndCast) {\n";
      os << tab << TAB << TAB <<  "std::cerr << \"WARNING: You already have a cast match of predicate\" << R\"(" << predicate << ")\";\n";
      os << tab << TAB << TAB <<  "assert(0);\n";
      os << tab << TAB <<  "}; \n";
      os << tab << TAB <<  "matchPredicateAndCast = true; \n";
      if (mach_type == MachineType::GPU)
      {
	 //os << tab << TAB <<  "bool sizeIncreased = false; \n";
         ////NOTE: For each subarray data members
         //--> bool sizeIncreased_um_inputs  = false; 
	 std::map<std::string, bool> used_name;
	 for (it2 = _interfaces.begin(); it2 != end2; ++it2) {
	    for (auto it = it2->second->getMappings().begin(); it != it2->second->getMappings().end(); ++it) {
	       std::string name(it->getDataType()->getName());
	       if (it->getDataType()->isArray() and used_name.count(name) == 0)
	       {
		  os << tab << TAB <<  "bool sizeIncreased_" << PREFIX_MEMBERNAME << it->getDataType()->getName() << " = false; \n";
		  used_name[name] = true;
	       }
	    }
	 }
	 for (auto it = _psetMappings.begin(); it != _psetMappings.end(); ++it) {
	    std::string name(it->second->getName());
	    if (it->second->isArray() and used_name.count(name) == 0)
	    {
	       used_name[name] = true;
	       os << tab << TAB <<  "bool sizeIncreased_" << PREFIX_MEMBERNAME << it->second->getName() << " = false; \n";
	    }
	 }
      }

      for (it = requiredIncreases.begin(); it != end; ++it) {
	 std::string sizeName = PREFIX + *it + "Size";
	 os << tab << TAB << "int " << sizeName << " = " << *it 
	    << ".size();\n"
	    << tab << TAB << *it << ".increase();\n";
      }

      os <<  interfaceToMemberCodes
	 <<  psetToMemberCodes;
      os << tab <<  "} \n";
      //os << tab <<  "match = match || matchLocal; \n";
   }else{
      // code not in addPreNode_Dummy(..)
      std::set<std::string> interfaceNames = getInterfaceNames();
      std::set<std::string>::iterator it, end = interfaceNames.end();
      os << tab << "bool castMatchLocal = true;\n";
      os << tab << "noPredicateMatch = true;\n";
      for (it = interfaceNames.begin(); it != end; ++it) {
	 os << tab << "if (" << PREFIX << *it << "Ptr == 0) {\n"
	    << "#if !defined(NOWARNING_DYNAMICCAST) \n"
	    << tab << TAB << "std::cerr << \"Dynamic Cast of "
	    << *it << " failed in " << name
	    << "\" << std::endl;\n"
	    << "#endif\n"
	    //<< tab << TAB << "exit(-1);\n"
	    << tab << TAB << "castMatchLocal = false;\n"
	    << tab << "}\n"
	    << "\n";
      }

      MemberContainer<InterfaceToMember>::const_iterator it2, 
	 end2 = _interfaces.end();

      std::set<std::string> requiredIncreases;
      std::string interfaceToMemberCodes;
      std::string psetToMemberCodes;
      std::string subtab = tab + TAB;
      for (it2 = _interfaces.begin(); it2 != end2; ++it2) {
	 interfaceToMemberCodes += 
	    it2->second->getInterfaceToMemberCode(subtab, requiredIncreases, mach_type);
      }
      psetToMemberCodes = _psetMappings.getPSetToMemberCode(subtab, 
	    requiredIncreases, mach_type);

      end = requiredIncreases.end();
      os << tab <<  "if (castMatchLocal) { \n";
      os << tab << TAB <<  "if (matchPredicateAndCast) {\n";
      os << tab << TAB << TAB <<  "std::cerr << \"WARNING: You already have a cast match of predicate\" << R\"(" << predicate << ")\";\n";
      os << tab << TAB << TAB <<  "assert(0);\n";
      os << tab << TAB <<  "}; \n";
      os << tab << TAB <<  "matchPredicateAndCast = true; \n";
      if (mach_type == MachineType::GPU)
      {
	 os << tab << TAB <<  "bool sizeIncreased = false; \n";
      }

      for (it = requiredIncreases.begin(); it != end; ++it) {
	 std::string sizeName = PREFIX + *it + "Size";
	 os << tab << TAB << "int " << sizeName << " = " << *it 
	    << ".size();\n"
	    << tab << TAB << *it << ".increase();\n";
      }

      os <<  interfaceToMemberCodes
	 <<  psetToMemberCodes;
      os << tab <<  "} \n";
      //os << tab <<  "match = match || matchLocal; \n";
   }
   return os.str();   
}

std::string Connection::getCommonConnectionCodeAlternativeInterfaceSet(const std::string& tab,
						const std::string& name) const
{
   std::ostringstream os;

   std::set<std::string> interfaceNames = getInterfaceNames();
   std::set<std::string>::iterator it, end = interfaceNames.end();
	 os << tab << "match = false;\n";
	 os << tab << "matchLocal = true;\n";
   for (it = interfaceNames.begin(); it != end; ++it) {
      os << tab << "if (" << PREFIX << *it << "Ptr == 0) {\n"
	 << "#if !defined(NOWARNING_DYNAMICCAST) \n"
	 << tab << TAB << "std::cerr << \"Dynamic Cast of "
	 << *it << " failed in " << name
	 << "\" << std::endl;\n"
	 << "#endif\n"
	 //<< tab << TAB << "exit(-1);\n"
	 << tab << TAB << "matchLocal = false;\n"
	 << tab << "}\n"
	 << "\n";
   }

   MemberContainer<InterfaceToMember>::const_iterator it2, 
      end2 = _interfaces.end();
   
   std::set<std::string> requiredIncreases;
   std::string interfaceToMemberCodes;
   std::string psetToMemberCodes;
   for (it2 = _interfaces.begin(); it2 != end2; ++it2) {
      interfaceToMemberCodes += 
	 it2->second->getInterfaceToMemberCode(tab, requiredIncreases);
   }
   psetToMemberCodes = _psetMappings.getPSetToMemberCode(tab, 
							 requiredIncreases);

   end = requiredIncreases.end();
   os << tab <<  "if (matchLocal) { \n";
   os << tab << TAB <<  "match = true; \n";

   os << tab << TAB <<"//for the same inAttr, only enable one path of connection \n";
   os << tab << TAB <<"//from one node to another\n";
   //os << tab << TAB << predicate <<"\n";
   os << tab << TAB		<< "if (map_inAttr.count(CG_castedPSet->identifier) == 0)\n";
   os << tab << TAB << TAB <<"map_inAttr[CG_castedPSet->identifier] = 1;\n";
   os << tab << TAB << "else\n";
   os << tab << TAB << TAB << "map_inAttr[CG_castedPSet->identifier] += 1;\n";

   for (it = requiredIncreases.begin(); it != end; ++it) {
      std::string sizeName = PREFIX + *it + "Size";
      os << tab << TAB << "int " << sizeName << " = " << *it 
	 << ".size();\n"
	 << tab << TAB << *it << ".increase();\n";
   }
   

   os << tab << TAB << interfaceToMemberCodes
      << tab << TAB << psetToMemberCodes;
	 os << tab <<  "} \n";
	 //os << tab <<  "match = match || matchLocal; \n";
   return os.str();   
}

void Connection::addInterfaceHeaders(Class& instance) const
{
   MemberContainer<InterfaceToMember>::const_iterator it, 
      end = _interfaces.end();
   for (it = _interfaces.begin(); it != end; ++it) {
      instance.addHeader("\"" + it->first + ".h\"");
   }
}

void Connection::setPSetMappingsPSet(ConnectionCCBase* cc)
{
   if (_directionType == _PRE) {
      _psetMappings.setPSet(cc->getInAttrPSet());
   } else { // _POST
      _psetMappings.setPSet(cc->getOutAttrPSet());
   }
}

std::string Connection::getTypeString() const
{
   std::string retVal = "Connection ";
   retVal += getStringForDirectionType();
   retVal += " ";
   retVal += getStringForComponentType();
   return retVal;
}

std::string Connection::getStringForComponentType() const
{
   return getStringForComponentType(_componentType);
}

std::string Connection::getStringForDirectionType() const
{
   return getStringForDirectionType(_directionType);
}

std::string Connection::getStringForComponentType(ComponentType type)
{
   switch(type) {
   case _EDGE:
      return "Edge";
   case _NODE:
      return "Node";
   case _CONSTANT:
      return "Constant";
   case _VARIABLE:
      return "Variable";
   }
   assert(0);
   return "";
}

std::string Connection::getParameterNameForComponentType(ComponentType type)
{
   switch(type) {
   case _EDGE:
      return PREFIX + "edge";
   case _NODE:
      return PREFIX + "node";
   case _CONSTANT:
      return PREFIX + "constant";
   case _VARIABLE:
      return PREFIX + "variable";
   }
   assert(0);
   return "";
}

std::string Connection::getParametersForComponentType(ComponentType type)
{
   switch(type) {
   case _EDGE:
      return "0, " + PREFIX + "edge " + ", 0, 0";
   case _NODE:
      return PREFIX + "node " + ",0 , 0, 0";
   case _CONSTANT:
      return "0, 0, 0, " + PREFIX + "constant";
   case _VARIABLE:
      return "0, 0, " + PREFIX + "variable " + ", 0";
   }
   assert(0);
   return "";
}

std::string Connection::getStringForDirectionType(DirectionType type)
{
   if (type == _PRE) {
      return "Pre";
   } else {
      return "Post";
   }   
}

std::string Connection::getParametersForDirectionType(DirectionType type)
{
   if (type == _PRE) {
      return INATTRPSETNAME + ", 0";
   } else {
      return "0, " + OUTATTRPSETNAME;
   }   
}

void Connection::addInterfaceToMember(std::unique_ptr<InterfaceToMember>&& im) 
{
   _interfaces.addMemberToFront(im->getInterface()->getName(), std::move(im));
}


void Connection::addMappingToInterface(
   const std::string& interface, const std::string& interfaceMember,
   const std::string& typeStr, std::unique_ptr<DataType>&& dtToInsert)
{
   InterfaceToMember* curIm;
   try {
      curIm = _interfaces.getMember(interface);
   } catch (NotFoundException& e) {
      std::cerr << "in " << typeStr << ", interface " << e.getError() << std::endl;
      e.setError("");
      throw;
   }
   try {
      curIm->addMapping(interfaceMember, std::move(dtToInsert));
   } catch(GeneralException& e) {
      std::ostringstream os;
      os << "in " << typeStr << ", " << e.getError();
      throw ConnectionException(os.str());
   }   
}
